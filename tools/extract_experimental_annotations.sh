in_file=$1
out_file=$2

awk '$7 ~ /EXP|IDA|IPI|IMP|IGI|IEP|TAS|IC/{print $0}' ${in_file} > ${out_file}
