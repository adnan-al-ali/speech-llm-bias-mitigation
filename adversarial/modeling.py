from transformers import Qwen2AudioForConditionalGeneration, VoxtralForConditionalGeneration
import torch
from gradient_reversal import GradientReversalLayer
from transformers.models.qwen2_audio.modeling_qwen2_audio import Cache, Unpack, TransformersKwargs, create_bidirectional_mask, is_torchdynamo_compiling, logger, torch_compilable_check, Qwen2AudioCausalLMOutputWithPast, nn
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration, BaseModelOutputWithDeepstackFeatures
from nemotron.modeling import NemotronH_Nano_Omni_Reasoning_V3
from trainable_module import TrainableModule
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, Unpack
import numpy as np

class CosineWithWarmup(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, args, train):
        self._warmup_steps = args.epochs_warmup * len(train)
        self._decay_steps = args.lr_decay and (args.epochs - args.epochs_warmup) * len(train)
        super().__init__(optimizer, self.current_learning_rate)

    def current_learning_rate(self, step):
        assert step >= 0 and (not self._decay_steps or step <= self._warmup_steps + self._decay_steps)
        return step / self._warmup_steps if step < self._warmup_steps else \
            0.5 * (1 + torch.cos(torch.pi * (step - self._warmup_steps) / self._decay_steps)) if self._decay_steps else 1


class QwenWithReversedGenderHead(TrainableModule):
    def __init__(self, model: Qwen2AudioForConditionalGeneration, gradient_reversal_alpha=1.0):
        super().__init__()
        self.model = model
        self.gradient_reversal_layer = GradientReversalLayer(alpha=gradient_reversal_alpha)
        self.gender_head = torch.nn.Linear(model.config.audio_config.d_model, 1, device=model.device, dtype=model.dtype)
        self.keras_init(self.gender_head)
        self.encoder_frozen = False

        self.freeze_qwen_weights()
    
    def __move_to_lm_device(self, tensor):
        if tensor is not None:
            return tensor.to(self.model.language_model.device)
        return None

    def forward(self,
        input_ids: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],):
        
        # region Copied from Qwen2AudioForConditionalGeneration.forward
        target_device = self.model.audio_tower.device

        if input_features is not None:
            input_features = input_features.to(target_device)
            feature_attention_mask = feature_attention_mask.to(target_device)

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

            # 2. Merge text and audios
            if input_features is not None and input_ids.shape[1] != 1:
                audio_feat_lengths, audio_output_lengths = self.model.audio_tower._get_feat_extract_output_lengths(
                    feature_attention_mask.sum(-1)
                )
                batch_size, _, max_mel_seq_len = input_features.shape
                max_seq_len = (max_mel_seq_len - 2) // 2 + 1
                # Create a sequence tensor of shape (batch_size, max_seq_len)
                seq_range = (
                    torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device)
                    .unsqueeze(0)
                    .expand(batch_size, max_seq_len)
                )
                lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
                # Create mask
                padding_mask = seq_range >= lengths_expand
                audio_attention_mask_2d = (~padding_mask).to(dtype=torch.long, device=audio_feat_lengths.device)

                dummy_embeds = torch.zeros(
                    (batch_size, max_seq_len, 1),
                    dtype=inputs_embeds.dtype,
                    device=inputs_embeds.device,
                )

                audio_attention_mask = create_bidirectional_mask(
                    config=self.model.audio_tower.config,
                    inputs_embeds=dummy_embeds,
                    attention_mask=audio_attention_mask_2d,
                )

                audio_outputs = self.model.audio_tower(input_features, attention_mask=audio_attention_mask)
                selected_audio_feature = audio_outputs.last_hidden_state
                audio_features = self.model.multi_modal_projector(selected_audio_feature)

                # if we have consecutive audio tokens, then it means we expanded input_ids in processing
                audio_tokens = input_ids == self.model.config.audio_token_id
                legacy_processing = (audio_tokens[:, :-1] & audio_tokens[:, 1:]).sum() == 0

                if not is_torchdynamo_compiling() and legacy_processing:
                    logger.warning_once(
                        "Expanding inputs for audio tokens in Qwen2Audio should be done in processing."
                    )
                    inputs_embeds, attention_mask, labels, position_ids, _ = self.model._merge_input_ids_with_audio_features(
                        audio_features, audio_output_lengths, inputs_embeds, input_ids, attention_mask, labels
                    )
                else:
                    num_audios, max_audio_tokens, embed_dim = audio_features.shape
                    audio_features_mask = torch.arange(max_audio_tokens, device=audio_output_lengths.device)[None, :]
                    audio_features_mask = audio_features_mask < audio_output_lengths[:, None]
                    audio_features = audio_features[audio_features_mask]

                    n_audio_tokens = (input_ids == self.model.config.audio_token_id).sum().item()
                    n_audio_features = audio_features.shape[0]
                    torch_compilable_check(
                        n_audio_tokens == n_audio_features,
                        f"Audio features and audio tokens do not match, tokens: {n_audio_tokens}, features: {n_audio_features}",
                    )
                    special_audio_mask = (input_ids == self.model.config.audio_token_id).to(inputs_embeds.device)
                    special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds)
                    audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(special_audio_mask, audio_features)

        outputs = self.model.language_model(
            attention_mask=self.__move_to_lm_device(attention_mask),
            position_ids=self.__move_to_lm_device(position_ids),
            past_key_values=self.__move_to_lm_device(past_key_values),
            inputs_embeds=self.__move_to_lm_device(inputs_embeds),
            use_cache=use_cache,
            **kwargs,
        )

        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        

        llm_output = Qwen2AudioCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            attention_mask=attention_mask,
        )
        # endregion

        audio_features_for_adversarial = audio_outputs.last_hidden_state.mean(dim=1)  # (batch_size, d_model)
        reversed_features = self.gradient_reversal_layer(audio_features_for_adversarial)
        gender_logits = self.gender_head(reversed_features)  # (batch_size, 1)

        return (torch.permute(logits, (0, 2, 1)), gender_logits)

    def freeze_qwen_weights(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.encoder_frozen = True
    
    def unfreeze_encoder_weights(self):
        for param in self.model.audio_tower.parameters():
            param.requires_grad = True
        self.encoder_frozen = False
    
    def save_weights(self, path):
        """Save the model weights to the given path."""
        if self.encoder_frozen:
            return
        state_dict = self.model.audio_tower.state_dict()
        torch.save(state_dict, path)
        

class Qwen3OmniWithReversedGenderHead(TrainableModule):
    def __init__(self, model: Qwen3OmniMoeThinkerForConditionalGeneration, gradient_reversal_alpha=1.0):
        super().__init__()
        self.model = model
        self.gradient_reversal_layer = GradientReversalLayer(alpha=gradient_reversal_alpha)
        self.gender_head = torch.nn.Linear(model.config.audio_config.output_dim, 1, device=model.device, dtype=model.dtype)
        self.keras_init(self.gender_head)
        self.encoder_frozen = False

        self.freeze_qwen_weights()
    
    def __move_to_lm_device(self, tensor):
        if tensor is not None:
            return tensor.to(self.model.language_model.device)
        return None

    def forward(
        self,
        input_ids=None,
        input_features=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        feature_attention_mask=None,
        audio_feature_lengths=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        rope_deltas=None,
        labels=None,
        use_cache=None,
        output_router_logits: bool | None = None,
        use_audio_in_video=None,
        video_second_per_grid=None,
        **kwargs,
    ):
        # region Copied from Qwen3OmniMoeThinkerForConditionalGeneration.forward
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.model.config.text_config.output_router_logits
        )

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        visual_embeds_multiscale = None
        visual_pos_masks = None
        image_mask, video_mask = None, None
        # 2. Merge text , audios , image and video
        if input_features is not None:
            audio_features = self.model.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
                return_dict=True,
            ).last_hidden_state
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            _, _, audio_mask = self.model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        if pixel_values is not None:
            image_outputs: BaseModelOutputWithDeepstackFeatures = self.model.get_image_features(
                pixel_values, image_grid_thw, return_dict=True
            )
            image_embeds = image_outputs.pooler_output
            image_embeds_multiscale = image_outputs.deepstack_features
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _, _ = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_outputs: BaseModelOutputWithDeepstackFeatures = self.model.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True
            )
            video_embeds = video_outputs.pooler_output
            video_embeds_multiscale = video_outputs.deepstack_features
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask, _ = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask | image_mask
            visual_embeds_multiscale_joint = ()
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(image_embeds_multiscale, video_embeds_multiscale):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1])
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                visual_embeds_multiscale_joint = visual_embeds_multiscale_joint + (embed_joint,)
            visual_embeds_multiscale = visual_embeds_multiscale_joint
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_embeds_multiscale = image_embeds_multiscale
            visual_pos_masks = image_mask
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_embeds_multiscale = video_embeds_multiscale
            visual_pos_masks = video_mask

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None

        if attention_mask is not None and position_ids is None:
            past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
            if past_key_values_length == 0 or self.model.rope_deltas is None:
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.model.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                rope_deltas = rope_deltas - delta0
                self.model.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = (past_key_values_length + self.model.rope_deltas).to(input_ids.device)
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            deepstack_visual_embeds=visual_embeds_multiscale,
            visual_pos_masks=visual_pos_masks,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.model.lm_head(hidden_states)
        # endregion

        audio_mask_matrix = audio_mask.all(dim=-1)  # (batch_size, seq_len)
        audio_lengths = audio_mask_matrix.sum(dim=-1)  # (batch_size)

        sequences = torch.split(audio_features, audio_lengths.cpu().tolist())
        padded_audio_features = pad_sequence(
            sequences,
            batch_first=True,
            padding_value=0.0
        ) # (max_audio_len, batch_size, d_model)
        audio_attention_mask = torch.arange(padded_audio_features.size(1), device=padded_audio_features.device)[None, :] < audio_lengths[:, None]

        masked_mean = (padded_audio_features * audio_attention_mask.unsqueeze(-1)).sum(dim=1) / audio_attention_mask.sum(dim=1).unsqueeze(-1)  # (batch_size, d_model)
        audio_features_for_adversarial = masked_mean  # (batch_size, d_model)
        reversed_features = self.gradient_reversal_layer(audio_features_for_adversarial)
        gender_logits = self.gender_head(reversed_features)  # (batch_size, 1)

        return (torch.permute(logits, (0, 2, 1)), gender_logits)
    
    def freeze_qwen_weights(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.encoder_frozen = True
    
    def unfreeze_encoder_weights(self):
        for param in self.model.audio_tower.parameters():
            param.requires_grad = True
        self.encoder_frozen = False
    
    def save_weights(self, path):
        """Save the model weights to the given path."""
        if self.encoder_frozen:
            return
        state_dict = self.model.audio_tower.state_dict()
        torch.save(state_dict, path)


class VoxtralWithReversedGenderHead(TrainableModule):
    def __init__(self, model: VoxtralForConditionalGeneration, gradient_reversal_alpha=1.0):
        super().__init__()
        self.model = model
        self.gradient_reversal_layer = GradientReversalLayer(alpha=gradient_reversal_alpha)
        self.gender_head = torch.nn.Linear(model.config.audio_config.hidden_size, 1, device=model.device, dtype=model.dtype)
        self.keras_init(self.gender_head)
        self.encoder_frozen = False

        self.freeze_qwen_weights()


    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if input_features is not None and input_ids is not None:
            audio_embeds_raw = self.model.get_audio_features(input_features, return_dict=True)
            audio_embeds = audio_embeds_raw.pooler_output
        
            # replace text-audio token placeholders with audio embeddings
            audio_token_mask = (input_ids == self.model.config.audio_token_id).unsqueeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_token_mask.to(inputs_embeds.device), audio_embeds.to(inputs_embeds.device)
            )

        outputs = self.model.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        audio_features = audio_embeds_raw.last_hidden_state # (batch_size, seq_len, d_model)
        # > Voxtral does not support masking of the `input_features`, this argument is preserved for compatibility,
        # > but it is not used. By default the silence in the input log mel spectrogram are ignored.
        masked_mean = audio_features.mean(dim=1)  # (batch_size, d_model)
        reversed_features = self.gradient_reversal_layer(masked_mean)
        gender_logits = self.gender_head(reversed_features)  # (batch_size, 1)
        return (torch.permute(outputs.logits, (0, 2, 1)), gender_logits)


        masked_mean = (padded_audio_features * audio_attention_mask.unsqueeze(-1)).sum(dim=1) / audio_attention_mask.sum(dim=1).unsqueeze(-1)  # (batch_size, d_model)
        audio_features_for_adversarial = masked_mean  # (batch_size, d_model)
        reversed_features = self.gradient_reversal_layer(audio_features_for_adversarial)
        gender_logits = self.gender_head(reversed_features)  # (batch_size, 1)

        return (torch.permute(logits, (0, 2, 1)), gender_logits)
    
    def freeze_qwen_weights(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.encoder_frozen = True
    
    def unfreeze_encoder_weights(self):
        for param in self.model.audio_tower.parameters():
            param.requires_grad = True
        self.encoder_frozen = False
    
    def save_weights(self, path):
        """Save the model weights to the given path."""
        if self.encoder_frozen:
            return
        state_dict = self.model.audio_tower.state_dict()
        torch.save(state_dict, path)

class NemotronWithReversedGenderHead(TrainableModule):
    def __init__(self, model: NemotronH_Nano_Omni_Reasoning_V3, gradient_reversal_alpha=1.0):
        super().__init__()
        self.model = model
        self.gradient_reversal_layer = GradientReversalLayer(alpha=gradient_reversal_alpha)
        self.gender_head = torch.nn.Linear(model.config.sound_config.hidden_size, 1, device=model.device, dtype=model.dtype)
        self.keras_init(self.gender_head)
        self.encoder_frozen = False

        self.freeze_backbone_weights()

    def extract_sound_feature(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract and project sound features from audio input.
        
        Args:
            input_features: Mel spectrogram features [batch, seq_len, feature_dim]
            attention_mask: Optional attention mask [batch, seq_len]
            
        Returns:
            Sound embeddings projected to LLM hidden size [batch, encoded_seq_len, llm_hidden_size]
        """
        if self.model.sound_encoder is None:
            raise RuntimeError("Sound encoder not initialized. Check if sound_config is provided.")
        
        # Encode audio features
        sound_embeds_raw, sound_attention_mask = self.model.sound_encoder(input_features, attention_mask, output_attention_mask=True)
        sound_embeds_raw = sound_embeds_raw.to(dtype=torch.bfloat16)
        
        # Project to LLM hidden size
        sound_embeds = self.model.sound_projection(sound_embeds_raw)
        
        return sound_embeds, sound_embeds_raw, sound_attention_mask

    def forward(
            self,
            pixel_values: torch.FloatTensor = None,
            input_ids: torch.LongTensor = None,
            input_features: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[list[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            inputs_embeds = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        is_raw_waveform = False
        if isinstance(input_features, (list, tuple)):
            # List of audio clips (waveforms)
            is_raw_waveform = True
            waveforms = input_features
        elif isinstance(input_features, np.ndarray):
            # Numpy array - raw waveform
            is_raw_waveform = True
            waveforms = [input_features.squeeze()] if input_features.ndim > 1 else [input_features]
        elif isinstance(input_features, torch.Tensor):
            if input_features.dim() == 1:
                # 1D tensor - single raw waveform
                is_raw_waveform = True
                waveforms = [input_features.cpu().numpy()]
            elif input_features.dim() == 2:
                # 2D tensor [batch, samples] - batched raw waveforms
                is_raw_waveform = True
                waveforms = [clip.cpu().numpy() for clip in input_features]
            else:
                # 3D tensor [batch, seq_len, num_mel_bins] - pre-extracted features
                is_raw_waveform = False
        else:
            is_raw_waveform = False
        
        if is_raw_waveform:
            # Convert raw waveforms to mel spectrogram features
            audio_inputs = self.model.sound_feature_extractor(
                waveforms,
                sampling_rate=self.model.sound_feature_extractor.sampling_rate,
                return_tensors="pt",
            )
            sound_input_features = audio_inputs.input_features
            sound_attention_mask = audio_inputs.get("attention_mask", None)
        else:
            # Already extracted features
            sound_input_features = input_features
            sound_attention_mask = None
        
        # Move to correct device and dtype
        target_device = self.model.sound_encoder.encoder.subsampling.linear.weight.device
        target_dtype = self.model.language_model.config.torch_dtype
        
        sound_input_features = sound_input_features.to(dtype=target_dtype, device=target_device)
        if sound_attention_mask is not None:
            sound_attention_mask = sound_attention_mask.to(device=target_device)
        
        sound_embeds, sound_embeds_raw, encoded_sound_attention_mask = self.extract_sound_feature(sound_input_features, sound_attention_mask)

        if inputs_embeds is None:
            inputs_embeds = self.model.language_model.get_input_embeddings()(input_ids)

        if image_flags is not None:
            image_flags = image_flags.squeeze(-1)

        B, N, C = inputs_embeds.shape
        inputs_embeds = inputs_embeds.reshape(B * N, C)
        input_ids_copy = input_ids.reshape(B * N)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.model.img_context_token_id)

        if pixel_values is not None:
            vit_batch_size = pixel_values.shape[0]
            vit_embeds = self.model.extract_feature(pixel_values)

            del pixel_values

            if torch.distributed.get_rank() == 0:
                print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

            vit_embeds = vit_embeds[image_flags == 1]
            try:
                inputs_embeds[selected] = inputs_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            except Exception as e:
                vit_embeds = vit_embeds.reshape(-1, C)
                print(f'warning: {e}, inputs_embeds[selected].shape={inputs_embeds[selected].shape}, '
                    f'vit_embeds.shape={vit_embeds.shape}')
                n_token = selected.sum()
                inputs_embeds[selected] = inputs_embeds[selected] * 0.0 + vit_embeds[:n_token]

            del vit_embeds

        # Replace sound tokens with sound embeddings
        # if sound_embeds is not None and self.model.sound_context_token_id is not None:
        #     sound_mask = (input_ids == self.model.sound_context_token_id)
        #     assert sound_mask.sum() != 0, "No sound tokens found in input_ids"
        #     # inputs_embeds[sound_mask] = sound_embeds.reshape(-1, C).to(inputs_embeds.device, inputs_embeds.dtype)
        #     inputs_embeds[sound_mask] = sound_embeds[encoded_sound_attention_mask.bool()].to(inputs_embeds.device, inputs_embeds.dtype)

        if sound_embeds is not None and self.model.sound_context_token_id is not None:
            sound_mask = (input_ids_copy == self.model.sound_context_token_id)
            assert sound_mask.sum() != 0, "No sound tokens found in input_ids"
            if sound_embeds.dim() == 3 and sound_embeds.shape[0] > 1 and sound_attention_mask is not None:
                # `attention_mask.sum() = L_i // hop` per row, but
                # `ParakeetFeatureExtractor` pads each row to `1 + L_i // hop`
                # mel frames in single-call mode (the trailing frame comes
                # from STFT center padding) — and the existing batch=1 path
                # consumes that frame's embed too. Add 1 here to match.
                natural_input_lengths = sound_attention_mask.sum(-1) + 1
                output_lengths = self.model.sound_encoder.encoder._get_subsampling_output_length(natural_input_lengths)
                flat = torch.cat(
                    [sound_embeds[i, : int(n)] for i, n in enumerate(output_lengths.tolist())],
                    dim=0,
                )
            else:
                flat = sound_embeds.reshape(-1, C)
            assert sound_mask.sum().item() == flat.shape[0], (
                f"sound token count ({sound_mask.sum().item()}) != encoder output count ({flat.shape[0]})"
            )
            inputs_embeds[sound_mask] = flat.to(inputs_embeds.device, inputs_embeds.dtype)

        inputs_embeds = inputs_embeds.reshape(B, N, C)

        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        masked_sound_embeds = sound_embeds_raw * encoded_sound_attention_mask.unsqueeze(-1)
        sound_embeds_mean = masked_sound_embeds.sum(dim=1) / encoded_sound_attention_mask.sum(dim=1).unsqueeze(-1)
        reversed_features = self.gradient_reversal_layer(sound_embeds_mean)
        gender_logits = self.gender_head(reversed_features.to(self.gender_head.weight.device))  # (batch_size, 1)

        return (torch.permute(logits, (0, 2, 1)), gender_logits)

    def freeze_backbone_weights(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.encoder_frozen = True

    def unfreeze_encoder_weights(self):
        for param in self.model.sound_encoder.parameters():
            param.requires_grad = True
        self.encoder_frozen = False
    
    def save_weights(self, path):
        """Save the model weights to the given path."""
        if self.encoder_frozen:
            return
        state_dict = self.model.sound_encoder.state_dict()
        torch.save(state_dict, path)