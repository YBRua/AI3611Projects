准备好环境后, 跑 baseline (注意将环境初始化部分改成自己的设置):

```bash
sbatch run.sh
```


注: 
1. evaluate.py 用于计算指标，预测结果 `prediction.json` 写成这样的形式:
```json
[
    {
        "img_id":"1386964743_9e80d96b05.jpg",
        "prediction":[
            "young boy is standing in a field of grass"
        ]
    },
    {
        "img_id":"3523559027_a65619a34b.jpg",
        "prediction":[
            "young boy is standing in a field of grass"
        ]
    },
    ......
]
```
调用方法：
```bash
python evaluate.py --prediction_file prediction.json \
                   --reference_file /dssg/home/acct-stu/stu464/data/image_caption/caption.txt \
                   --output_file result.txt
```
2. 首次调用 SPICE 时需要下载一些包，可以直接从以下位置拷贝到对应的位置:
* /dssg/home/acct-stu/stu464/.conda/envs/pytorch/lib/python3.7/site-packages/pycocoevalcap/spice/spice-1.0.jar
* /dssg/home/acct-stu/stu464/.conda/envs/pytorch/lib/python3.7/site-packages/pycocoevalcap/spice/lib/stanford-corenlp-3.6.0.jar
* /dssg/home/acct-stu/stu464/.conda/envs/pytorch/lib/python3.7/site-packages/pycocoevalcap/spice/lib/stanford-corenlp-3.6.0-models.jar
