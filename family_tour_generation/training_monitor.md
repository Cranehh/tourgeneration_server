# Training Monitor — 训练状态检查流程

## 快速状态检查脚本

每次 session 开始时运行此脚本获取训练快照：

```bash
LOG="../train_500ep.log"
echo "=== Training Status @ $(date) ==="

# ── 1. 进程与 GPU ──
echo ""
echo "--- Process & GPU ---"
PROC_COUNT=$(ps aux | grep train_with_ss | grep -v grep | wc -l)
echo "Training processes: $PROC_COUNT"
if [ "$PROC_COUNT" -eq 0 ]; then
  echo "⚠ WARNING: Training process NOT running!"
fi
nvidia-smi | grep MiB | head -1 2>/dev/null || echo "nvidia-smi unavailable"

# ── 2. Train Loss (last 10) ──
echo ""
echo "--- Train Loss Trend (last 10 epochs) ---"
grep "INFO" "$LOG" | grep -E "Epoch [0-9]+ - Loss:" | grep -v "Val" | tail -10 | \
  awk '{ep=$7; loss=$10; gsub(",","",ep); gsub(",","",loss); printf "  Ep%-4s  %.4f\n", ep, loss}'

# ── 3. Val Loss (last 10) ──
echo ""
echo "--- Val Loss Trend (last 10 epochs) ---"
grep "INFO" "$LOG" | grep "Val Loss:" | tail -10 | \
  awk '{
    ep=$7; vl=$11; tf=$16
    gsub(",","",ep); gsub(",","",vl); gsub(",","",tf)
    printf "  Ep%-4s  ValLoss=%-8s  TFValLoss=%s\n", ep, vl, tf
  }'

# ── 4. gen_acc (last 5 records, logged every 5 epochs) ──
echo ""
echo "--- gen_acc Trend (last 5 records) ---"
printf "  %-6s %-8s %-8s %-8s %-8s %-8s %-8s %-8s\n" "Epoch" "start" "end" "purp" "mode" "driver" "joint" "dest"
printf "  %-6s %-8s %-8s %-8s %-8s %-8s %-8s %-8s\n" "-----" "------" "------" "------" "------" "------" "------" "------"
paste <(grep -B 50 "gen_acc_purpose" "$LOG" | grep "Val Loss:" | awk '{ep=$7; gsub(",","",ep); print ep}' | tail -5) \
      <(grep "INFO.*gen_acc" "$LOG" | grep -E "gen_acc_(start_time|end_time|purpose|mode|driver|joint|destination)" | \
        awk '/gen_acc_start_time/{st=$NF} /gen_acc_end_time/{et=$NF} /gen_acc_purpose/{pu=$NF}
             /gen_acc_mode/{mo=$NF} /gen_acc_driver/{dr=$NF} /gen_acc_joint/{jo=$NF}
             /gen_acc_destination/{de=$NF; print st, et, pu, mo, dr, jo, de}' | tail -5) | \
  awk '{printf "  %-6s %-8s %-8s %-8s %-8s %-8s %-8s %-8s\n", $1,$2,$3,$4,$5,$6,$7,$8}'

# ── 5. 异常检测 ──
echo ""
echo "--- Anomaly Detection ---"
BF_COUNT=$(grep -c 'backward failed' "$LOG" 2>/dev/null)
NAN_COUNT=$(grep -c 'loss is NaN' "$LOG" 2>/dev/null)
echo "backward failed (cumulative): $BF_COUNT"
echo "loss is NaN (cumulative):     $NAN_COUNT"
if [ "$NAN_COUNT" -gt 0 ]; then
  echo "⚠ WARNING: NaN loss detected! Check recent log entries."
fi

# backward failed 增速检查: 最近 10 个 epoch 的 backward failed 数
RECENT_BF=$(tail -20000 "$LOG" | grep -c 'backward failed' 2>/dev/null)
echo "backward failed (recent ~10ep): $RECENT_BF"
if [ "$RECENT_BF" -gt 20 ]; then
  echo "⚠ WARNING: High backward failure rate in recent epochs!"
fi

# ── 6. 历史最佳值对比 ──
echo ""
echo "--- Historical Best Values ---"
echo "Val Loss best:"
grep "INFO" "$LOG" | grep "Val Loss:" | awk '{ep=$7; vl=$11; gsub(",","",ep); gsub(",","",vl); print ep, vl}' | sort -k2 -n | head -3 | awk '{printf "  Ep%-4s  %s\n", $1, $2}'

echo "TF Val Loss best:"
grep "INFO" "$LOG" | grep "Val Loss:" | awk '{ep=$7; tf=$16; gsub(",","",ep); gsub(",","",tf); print ep, tf}' | sort -k2 -n | head -3 | awk '{printf "  Ep%-4s  %s\n", $1, $2}'

echo "gen_acc best per metric:"
paste <(grep -B 50 "gen_acc_purpose" "$LOG" | grep "Val Loss:" | awk '{ep=$7; gsub(",","",ep); print ep}') \
      <(grep "INFO.*gen_acc" "$LOG" | grep -E "gen_acc_(start_time|end_time|purpose|mode|driver|joint|destination)" | \
        awk '/gen_acc_start_time/{st=$NF} /gen_acc_end_time/{et=$NF} /gen_acc_purpose/{pu=$NF}
             /gen_acc_mode/{mo=$NF} /gen_acc_driver/{dr=$NF} /gen_acc_joint/{jo=$NF}
             /gen_acc_destination/{de=$NF; print st, et, pu, mo, dr, jo, de}') | \
  awk '{
    ep[NR]=$1; st[NR]=$2; et[NR]=$3; pu[NR]=$4; mo[NR]=$5; dr[NR]=$6; jo[NR]=$7; de[NR]=$8; n=NR
  }
  END {
    best_st=999; best_et=999; best_pu=0; best_mo=0; best_dr=0; best_jo=0; best_de=0
    for(i=1;i<=n;i++){
      if(st[i]+0<best_st && st[i]+0>0){best_st=st[i]+0; ep_st=ep[i]}
      if(et[i]+0<best_et && et[i]+0>0){best_et=et[i]+0; ep_et=ep[i]}
      if(pu[i]+0>best_pu){best_pu=pu[i]+0; ep_pu=ep[i]}
      if(mo[i]+0>best_mo){best_mo=mo[i]+0; ep_mo=ep[i]}
      if(dr[i]+0>best_dr){best_dr=dr[i]+0; ep_dr=ep[i]}
      if(jo[i]+0>best_jo){best_jo=jo[i]+0; ep_jo=ep[i]}
      if(de[i]+0>best_de){best_de=de[i]+0; ep_de=ep[i]}
    }
    printf "  start_time:   %.4f  (Ep %s)  [lower=better]\n", best_st, ep_st
    printf "  end_time:     %.4f  (Ep %s)  [lower=better]\n", best_et, ep_et
    printf "  purpose:      %.4f  (Ep %s)\n", best_pu, ep_pu
    printf "  mode:         %.4f  (Ep %s)\n", best_mo, ep_mo
    printf "  driver:       %.4f  (Ep %s)\n", best_dr, ep_dr
    printf "  joint:        %.4f  (Ep %s)\n", best_jo, ep_jo
    printf "  destination:  %.4f  (Ep %s)\n", best_de, ep_de
  }'

# ── 7. 过拟合检测 ──
echo ""
echo "--- Overfitting Check ---"
grep "INFO" "$LOG" | grep -E "(Epoch [0-9]+ - Loss:|Val Loss:)" | \
  awk '/Epoch [0-9]+ - Loss:/{ep=$7; tl=$10; gsub(",","",ep); gsub(",","",tl); train_loss=tl; train_ep=ep}
       /Val Loss:/{vep=$7; vl=$11; gsub(",","",vep); gsub(",","",vl);
         if(train_ep==vep){gap=train_loss-vl; printf "Ep%-4s  Train=%-8s Val=%-8s Gap=%.4f\n", vep, train_loss, vl, gap}}' | tail -8

# ── 8. 学习率检查 ──
echo ""
echo "--- Learning Rate (last 5) ---"
grep "INFO" "$LOG" | grep "lr:" | tail -5 | awk '{for(i=1;i<=NF;i++){if($i~"lr:"){print $i, $(i+1)}}}'

echo ""
echo "=== Check Complete ==="
```

## 异常判定阈值

| 检查项 | 正常范围 | 警告阈值 | 严重阈值 |
|--------|---------|---------|---------|
| 训练进程数 | >= 1 | 0 (进程崩溃) | — |
| Train Loss | 稳定下降 | 连续 3 epoch 上升 | 突增 >50% |
| Val Loss | ~1.0 附近波动 | >1.2 或连续 5 epoch 上升 | >1.5 |
| Train-Val Gap | <0.6 | >0.8 (过拟合) | >1.0 |
| gen_acc_purpose | >0.80 | <0.78 | <0.75 |
| gen_acc_mode | >0.55 | <0.52 | <0.50 |
| gen_acc_driver | >0.86 | <0.84 | <0.82 |
| gen_acc_joint | >0.92 | <0.90 | <0.85 |
| gen_acc_destination | >0.80 | <0.78 | <0.75 |
| backward failed (近10ep) | <10 | 10-20 | >20 |
| loss is NaN | 0 | 任何非零值 | — |
| GPU 温度 | <80°C | 80-85°C | >85°C |
| GPU 显存占用 | <90% | 90-95% | >95% |

## 历史最佳记录 (截至 Epoch 136, 2026-02-25)

### Loss

| 指标 | 最佳值 | Epoch |
|------|--------|-------|
| Val Loss | 0.9964 | 117 |
| TF Val Loss | 0.9654 | 117 |
| Train Loss | 1.5436 | 135 |

### gen_acc (每 5 epoch 记录)

| 指标 | 最佳值 | Epoch | 方向 |
|------|--------|-------|------|
| start_time | 0.3394 | 124 | lower=better |
| end_time | 0.3360 | 124 | lower=better |
| purpose | 0.8055 | 129 | higher=better |
| mode | 0.5640 | 134 | higher=better |
| driver | 0.8734 | 99 | higher=better |
| joint | 0.9629 | 0 | higher=better (注: Ep0 为初始高估, 稳定期最佳 0.9440 @ Ep114) |
| destination | 0.8063 | 129 | higher=better |

## 训练阶段记录

| 阶段 | Epoch 范围 | 特征 |
|------|-----------|------|
| 快速下降期 | 0–49 | Loss 急剧下降, gen_acc 快速提升 |
| 稳定改善期 | 50–90 | Loss 持续下降但速度放缓 |
| 平台期 | 90–136+ | Val Loss ~1.0 波动, gen_acc 变化微小 |

## gen_acc 说明

- gen_acc 每 5 个 epoch 记录一次 (Ep0, 4, 9, 14, 19, 24, ...)
- start_time/end_time 为误差指标 (MAE), **越低越好**
- 其余指标 (purpose, mode, driver, joint, destination) 为准确率, **越高越好**
- joint 在 Ep0 的 0.9629 是因为初始阶段模型倾向于预测多数类，不应作为真实基准
