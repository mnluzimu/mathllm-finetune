start_idx=$1
interval=$2

bash /mnt/cache/luzimu/mathllm-finetune/src/inference_tool/Meta-Llama-3-8B_tool_numinamath-cot-lce_gsm8k-math-81007_add-sys_pre-sftmath-related-code-owm-cc-en-10500_open-web-math-code-added_problem-solution_all_v2_1759958_WebInstructSub_2335220_code-added-09210036/infer_checkpoints.sh 9600 $start_idx $interval
bash /mnt/cache/luzimu/mathllm-finetune/src/inference_tool/Meta-Llama-3-8B_tool_numinamath-cot-lce_gsm8k-math-81007_add-sys_pre-sftmath-related-code-owm-cc-en-10500_open-web-math-code-added_problem-solution_all_v2_1759958_WebInstructSub_2335220_code-added-09210036/infer_checkpoints.sh 9800 $start_idx $interval
bash /mnt/cache/luzimu/mathllm-finetune/src/inference_tool/Meta-Llama-3-8B_tool_numinamath-cot-lce_gsm8k-math-81007_add-sys_pre-sftmath-related-code-owm-cc-en-10500_open-web-math-code-added_problem-solution_all_v2_1759958_WebInstructSub_2335220_code-added-09210036/infer_checkpoints.sh 10000 $start_idx $interval
