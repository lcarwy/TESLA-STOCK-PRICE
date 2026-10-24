# 特斯拉（TSLA）上市 16 年 · 3D K 线故事视频

仿照 B 站 @湾林Studio《CS饰品五年大盘涨跌幅可视化》的风格，用**按日收盘价**做的特斯拉竖屏股价视频。

成品：[`output/tsla_vertical.mp4`](output/tsla_vertical.mp4)（1080×1920 竖屏，30fps，约 94 秒，带合成配乐和音效）

## 视频结构

| 时间 | 内容 |
| --- | --- |
| 0–3.7s | 冷开场「黑色星期四」：2026.07.23 Q2 财报暴雷，1 小时 K 线，单日 −14.5%，闪电 + 碎裂特效 |
| 3.7–5.2s | ◀◀ 倒带回到 2010.06.29 上市首日 |
| 5.2–57.6s | 13 个章节（周K / 日K）：纳斯达克上市 → 漫长横盘 → Model S 引爆 → 六年原地踏步 → 绝地反击 → 疫情闪崩（第 N 天计数）→ 史诗级暴涨 → 历史新高 → 腰斩再腰斩 → 触底反弹 → 降价潮 → 大选行情 → 关税与抵制 → Robotaxi 行情 → 高位阴跌 |
| 57.6–70.2s | 放大到 1 小时 K：Q2 财报卡片 → 一觉醒来天塌了 → 上市以来单日跌幅 TOP 9 → 多杀多跌破 300 美元 → 「8 个月的涨幅，回吐 71%」 |
| 70.2–75.8s | 超跌反弹 |
| 75.8–84.2s | 对数坐标回顾 16 年全程：+32,743%（×328），章节标签钉在走势上 |
| 84.2–94.5s | 后续走势？（随机路径扇形） |

画面元素：顶部 HUD（实时价格、相对基准涨跌幅、滚动日期、周K/日K/1小时K 标签）、章节面板（利好/利空标签 + 标题 + 说明）、章节区间框（涨跌幅 + 天数）、事件标签、3D 透视发光 K 线与红绿光幕、星尘粒子、闪电/碎裂/金色烟花等。红涨绿跌。

## 数据

- `data/tsla_daily.csv`：2010-06-29 至 2026-10-23 每个交易日的 OHLCV（Yahoo Finance，已按 2020 年 1 拆 5、2022 年 1 拆 3 调整；特斯拉从未派息）。周K 由日线聚合。
- `data/tsla_hourly.csv`：最近约两年的 1 小时 K（Yahoo 的上限）。每天最后一根小时线的收盘价对齐到当日官方收盘价（Yahoo 的小时线不含收盘集合竞价）。
- 视频里的涨跌幅、天数、历史新高、单日跌幅排名、「五次腰斩」等数字全部由脚本从数据计算，不是手填的。
- Q2 2026 财报卡片里的营收、EPS、利润率、自由现金流、资本开支来自当时的新闻报道（Yahoo Finance、Electrek 等）。

## 重新生成

依赖：Python 3、ffmpeg，`pip install skia-python numpy pandas yfinance`（skia 需要系统库 `libegl1`）。

```bash
python3 scripts/fetch_daily.py          # 更新日线 / 小时线数据
scripts/fetch_fonts.sh                  # 下载字体（思源黑体 SC、Barlow Semi Condensed，均为 SIL OFL 开源授权）
python3 make_vertical.py --preview 30 60   # 导出几帧截图检查版式
python3 make_vertical.py                # 完整渲染 → output/tsla_vertical.mp4
```

- 分镜（每段时间轴、镜头、章节文案、事件、特效时间点）都在 `tsla_video/story.py`。
- 画面绘制：`tsla_video/chart.py`（3D K 线）、`tsla_video/scene.py`（HUD / 面板 / 叠加层）、`tsla_video/effects.py`（粒子特效）。
- 配乐：`soundtrack.py` 用 numpy 合成，鼓点、和声和音效按分镜时间点对齐，无版权问题。

> 历史表现不代表未来收益，视频内容仅供参考，不构成任何投资建议。
