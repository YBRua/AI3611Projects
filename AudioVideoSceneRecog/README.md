baseline 用到 openl3 提取的 audio 和 visual 特征，此步骤过程较长，已预先提取好放于 `/dssg/home/acct-stu/stu464/ai3611/av_scene_classify/data/feature`，基于此跑 baseline (注意将环境初始化部分改成自己的设置):

```bash
sbatch run.sh
```

注: eval_pred.py 用于计算指标，预测结果 `prediction.csv` 写成这样的形式 (分隔符为 `\t`):
```
aid     scene_pred      airport     bus     ......  tram
airport-lisbon-1175-44106   airport     0.9   0.000   ......  0.001
......
```
调用方法：
```bash
python eval_prediction.py --prediction prediction.csv \
                          --label /dssg/home/acct-stu/stu464/data/audio_visual_scenes/evaluation_setup/fold1_evaluate.csv
```

注: 如果有同学想自己安装 openl3 环境，需要走代理下载 openl3 模型文件，进行如下设置:
```bash
export http_proxy="http://127.0.0.1:10802"
export https_proxy="http://127.0.0.1:10802"
```
随后即可用 `pip install -r requirements_openl3.txt` 安装环境
