# Qwen2.5 MedExpQA 逐题对比

在仓库根目录运行下面两行即可本地查看项目页。

```bash
python scripts/build_qwen25_pages.py
python -m http.server 8765 --bind 127.0.0.1 --directory outputs/qwen25-pages
```

浏览器打开 [本地项目页](http://127.0.0.1:8765/)。构建脚本用仓库中的逐题预测和同一套评分逻辑计算页面数据。三个精选题的英文题干及选项保存在 `featured_cases.json`，来源与许可见页面页脚。[预测文件和评测设置](../release/qwen25-medexpqa/)
