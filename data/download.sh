export HF_HOME=/mnt/nvme0n1/ligong/collab/ruoyu/cache/huggingface
echo "HF_HOME: $HF_HOME"
cd /workspace/home/ligong/collab/ruoyu/dLLM-KL/data
python download_data.py --dataset MATH500
# python download_data.py --dataset MATH_train