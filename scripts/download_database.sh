#!/usr/bin/bash

readonly db_dir=${1}

for cmd in wget tar ; do
  if ! command -v "${cmd}" > /dev/null 2>&1; then
    echo "${cmd} is not installed. Please install it."
  fi
done

echo "Downloading databases to ${db_dir}"
mkdir -p "${db_dir}"
cd "${db_dir}"

readonly NAME=library
readonly SOURCE='http://yanglab.qd.sdu.edu.cn/trRosettaRNA/download'
echo "Start Downloading ${NAME}"
wget "${SOURCE}/${NAME}.tar.bz2"
echo "Uncompress ${NAME}"
tar -xjvf "${NAME}.tar.bz2"

echo "Complete"