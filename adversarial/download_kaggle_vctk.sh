#!/bin/bash
source .env

# Ensure kaggle.json exists
KAGGLE_DIR="${HOME}/.kaggle"
KAGGLE_JSON="${KAGGLE_DIR}/kaggle.json"

if [[ -f "${KAGGLE_JSON}" ]]; then
    echo "kaggle.json already exists at ${KAGGLE_JSON}"
else
    mkdir -p "${KAGGLE_DIR}"
    printf '{"username":"%s","key":"%s"}\n' "${KAGGLE_USERNAME}" "${KAGGLE_KEY}" > "${KAGGLE_JSON}"
    chmod 600 "${KAGGLE_JSON}"
fi


mkdir downloads || true
mkdir vctk-corpus || true
cd downloads/vctk-corpus
kaggle datasets download pratt3000/vctk-corpus --unzip
rm CV_English.tar.gz