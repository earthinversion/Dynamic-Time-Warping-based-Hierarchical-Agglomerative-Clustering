#!/bin/bash


# Declare a string array with type
# declare -a StringArray=("NAAO" "TUNH" "TAPO" "ERPN" "CHEN" "PEPU" "TAPE" "YENL" "DAJN" "DAWU" "NDHU" "")
declare -a StringArray=('PEPU' 'DAJN' 'NDHU' 'CHUN' 'SHUL' 'TUNH' 'DAWU' 'CHGO' 'YENL' 'SHAN' 'SOFN' 'TAPE' 'ERPN' 'CHEN' 'TAPO' 'SINL' 'LONT' 'JULI' 'JSUI' 'TTUN' 'NAAO' 'SPAO' 'MOTN' 'SLNP' 'WARO' 'SLIN' 'WULU')

# Read the array values with space
for stn in "${StringArray[@]}"; do
  ls ${stn}.COR
done