#!/usr/bin/env bash
     2	set -euo pipefail
     3	
     4	# Runs all 4 PLH worlds with multiple roles and then builds a combined dataset.
     5	# Default total interactions: 500
     6	# Usage:
     7	#   bash scripts/run_all_worlds_and_build_dataset.sh
     8	#   TOTAL=800 MODEL=qwen2.5:7b PROVIDER=ollama bash scripts/run_all_worlds_and_build_dataset.sh
     9	
    10	TOTAL="${TOTAL:-500}"
    11	PROVIDER="${PROVIDER:-ollama}"
    12	MODEL="${MODEL:-qwen2.5:7b}"
    13	MODE="${MODE:-live}"
    14	SLEEP="${SLEEP:-1.0}"
    15	OUTPUT_ROOT="${OUTPUT_ROOT:-data/logs/multi_world_500}"
    16	RUN_PREFIX="${RUN_PREFIX:-multiworld}"
    17	
    18	# Keep all runs in one combined output directory.
    19	mkdir -p "${OUTPUT_ROOT}"
    20	
    21	declare -a RUNS=(
    22	  "healthcare_world doctor"
    23	  "healthcare_world nurse"
    24	  "ecommerce_world user"
    25	  "ecommerce_world support"
    26	  "cloud_devops_world developer"
    27	  "cloud_devops_world admin"
    28	  "social_media_world user"
    29	  "social_media_world moderator"
    30	)
    31	
    32	run_count=${#RUNS[@]}
    33	base=$(( TOTAL / run_count ))
    34	remainder=$(( TOTAL % run_count ))
    35	
    36	printf 'Starting multi-world collection\n'
    37	printf '  total=%s mode=%s provider=%s model=%s\n' "$TOTAL" "$MODE" "$PROVIDER" "$MODEL"
    38	printf '  output=%s\n\n' "$OUTPUT_ROOT"
    39	
    40	for i in "${!RUNS[@]}"; do
    41	  world="$(awk '{print $1}' <<<"${RUNS[$i]}")"
    42	  role="$(awk '{print $2}' <<<"${RUNS[$i]}")"
    43	  n=$base
    44	  if [[ "$i" -lt "$remainder" ]]; then
    45	    n=$((n + 1))
    46	  fi
    47	
    48	  run_name="${RUN_PREFIX}_${world}_${role}"
    49	  echo "[$((i+1))/${run_count}] world=${world} role=${role} num=${n}"
    50	
    51	  python scripts/run_pipeline.py \
    52	    --world "$world" \
    53	    --mode "$MODE" \
    54	    --provider "$PROVIDER" \
    55	    --model "$MODEL" \
    56	    --role "$role" \
    57	    --num "$n" \
    58	    --sleep "$SLEEP" \
    59	    --output "$OUTPUT_ROOT" \
    60	    --run-name "$run_name"
    61	done
    62	
    63	echo
    64	printf 'Building combined dataset from %s\n' "$OUTPUT_ROOT"
    65	python -c "from src.pipeline.dataset_builder import DatasetBuilder; db=DatasetBuilder('${OUTPUT_ROOT}'); entries=db.filter(db.load_all()); train,val,test=db.split(entries); db.save_splits(train,val,test,'data/features'); print(f'Built splits from {len(entries)} entries into data/features')"
    66	
    67	echo 'Done.'
