# R14_QA2 Clean Train Execution Checklist

## Main Judgment

- `QA1` 在 `train` 中给出 `296` 条建议修正，已经足够支撑 `clean_train` 分支。
- 当前最优先的问题不是继续换模型，而是先统一 `train` 侧标注边界和漏标。
- 三类问题规模大致是：`missing-rivet=87`，`rust=202`，`scratch=153`。

## Fix Order

- 第一优先级：`missing_label`。先补高置信漏标，因为这类问题最可能直接限制召回和真实上限。
- 第二优先级：`oversized_box`。先统一框边界，减少过大背景框对定位学习的干扰。
- 第三优先级：`wrong_class` 和 `shifted_box`。这两类数量少于前两者，但会直接污染类别判别和高 IoU 指标。
- 修订顺序建议先 `train`，不改 `valid`。`valid` 只保留问题记录，不纳入这轮 clean_train 修改。

## Class Priority

- `missing-rivet` 先修顺序：
`missing_label` (`33` 条, `27` 张图) -> 优先核查是否漏标，若确认是真缺陷则补框并补类别。
  `oversized_box` (`8` 条, `7` 张图) -> 优先缩框，去掉过多背景，统一边界风格。
  `wrong_class` (`3` 条, `2` 张图) -> 优先核查类别定义，必要时改类而不是重画框。
- `rust` 先修顺序：
  `missing_label` (`52` 条, `45` 张图) -> 优先核查是否漏标，若确认是真缺陷则补框并补类别。
  `oversized_box` (`94` 条, `86` 张图) -> 优先缩框，去掉过多背景，统一边界风格。
  `wrong_class` (`6` 条, `6` 张图) -> 优先核查类别定义，必要时改类而不是重画框。
- `scratch` 先修顺序：
  `missing_label` (`44` 条, `40` 张图) -> 优先核查是否漏标，若确认是真缺陷则补框并补类别。
  `oversized_box` (`46` 条, `45` 张图) -> 优先缩框，去掉过多背景，统一边界风格。
  `wrong_class` (`6` 条, `6` 张图) -> 优先核查类别定义，必要时改类而不是重画框。

## Top Images

| rank | tier | image | issue_count | dominant_class | dominant_error | max_conf | first_action | vis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | sceneadd1100581_jpg.rf.08616b80961b07310c6ba054853afd59.jpg | 4 | rust | oversized_box | 0.618 | 优先缩框，去掉过多背景，统一边界风格。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0089_missing_label_sceneadd1100581_jpg.rf.08616b80961b07310c6ba054853afd59.jpg |
| 2 | A | image-033-2-_jpg.rf.aea620d54c387d9fdbe32f5cf507d750.jpg | 3 | rust | missing_label | 0.796 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0005_missing_label_image-033-2-_jpg.rf.aea620d54c387d9fdbe32f5cf507d750.jpg |
| 3 | A | sceneadd400241_jpg.rf.7246494984ab948d4c65793e33ae72e7.jpg | 3 | scratch | missing_label | 0.719 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0018_missing_label_sceneadd400241_jpg.rf.7246494984ab948d4c65793e33ae72e7.jpg |
| 4 | A | image-027-3-_jpg.rf.fa07abc92d7dc37e821ec91595c81519.jpg | 3 | missing-rivet | missing_label | 0.684 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0037_missing_label_image-027-3-_jpg.rf.fa07abc92d7dc37e821ec91595c81519.jpg |
| 5 | A | sceneadd701541_jpg.rf.338f574fddd9b5f963cfe0f50a796b5a.jpg | 3 | missing-rivet | missing_label | 0.650 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0058_missing_label_sceneadd701541_jpg.rf.338f574fddd9b5f963cfe0f50a796b5a.jpg |
| 6 | A | image-020-3-_jpg.rf.71e7c40c6add8a544d3bf5905386c76e.jpg | 3 | rust | missing_label | 0.613 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0091_missing_label_image-020-3-_jpg.rf.71e7c40c6add8a544d3bf5905386c76e.jpg |
| 7 | A | sceneadd1302141_jpg.rf.b02a07d348c6d97f136c485433da0bac.jpg | 3 | rust | missing_label | 0.689 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0034_missing_label_sceneadd1302141_jpg.rf.b02a07d348c6d97f136c485433da0bac.jpg |
| 8 | A | image-064_jpg.rf.7f8a7b3ac5e051c11db1fec3fb7c3bbb.jpg | 3 | scratch | missing_label | 0.692 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0030_missing_label_image-064_jpg.rf.7f8a7b3ac5e051c11db1fec3fb7c3bbb.jpg |
| 9 | A | image-059-1-_jpg.rf.c6eab1ec42e7574325669c1d6f782a9b.jpg | 3 | rust | missing_label | 0.532 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0157_missing_label_image-059-1-_jpg.rf.c6eab1ec42e7574325669c1d6f782a9b.jpg |
| 10 | A | sceneadd1100541_jpg.rf.a2590308da5ee198eee51c7053bd95f8.jpg | 3 | rust | oversized_box | 0.724 | 优先缩框，去掉过多背景，统一边界风格。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0016_missing_label_sceneadd1100541_jpg.rf.a2590308da5ee198eee51c7053bd95f8.jpg |
| 11 | B | sceneadd1300381_jpg.rf.ae70c9eadb0a6532b3f399a218b9c4da.jpg | 2 | missing-rivet | missing_label | 0.773 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0008_missing_label_sceneadd1300381_jpg.rf.ae70c9eadb0a6532b3f399a218b9c4da.jpg |
| 12 | B | sceneadd700321_jpg.rf.1e2a6c793d15dcb7b35da80268232056.jpg | 2 | scratch | missing_label | 0.730 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0015_missing_label_sceneadd700321_jpg.rf.1e2a6c793d15dcb7b35da80268232056.jpg |
| 13 | B | sceneadd302621_jpg.rf.f479c2b9ea7f3f14b04421b07b2d1606.jpg | 2 | rust | missing_label | 0.686 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0035_missing_label_sceneadd302621_jpg.rf.f479c2b9ea7f3f14b04421b07b2d1606.jpg |
| 14 | B | sceneadd401021_jpg.rf.ff6826e2a386f68c333109e7e0682a20.jpg | 2 | missing-rivet | missing_label | 0.693 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0029_missing_label_sceneadd401021_jpg.rf.ff6826e2a386f68c333109e7e0682a20.jpg |
| 15 | B | image-020-1-_jpg.rf.718e0f870b030ae77e4829fc024ede18.jpg | 2 | rust | missing_label | 0.658 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0054_missing_label_image-020-1-_jpg.rf.718e0f870b030ae77e4829fc024ede18.jpg |
| 16 | B | sceneadd1600341_jpg.rf.626aacfe3c15640fac0c87a315aa6574.jpg | 2 | missing-rivet | missing_label | 0.749 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0013_missing_label_sceneadd1600341_jpg.rf.626aacfe3c15640fac0c87a315aa6574.jpg |
| 17 | B | image-015-2-1-_jpg.rf.b171e786f011dcd9c824aae2bc892403.jpg | 2 | rust | missing_label | 0.691 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0031_missing_label_image-015-2-1-_jpg.rf.b171e786f011dcd9c824aae2bc892403.jpg |
| 18 | B | sceneadd1100521_jpg.rf.e87ffc7adc2925a7cc54323f23a52366.jpg | 3 | rust | oversized_box | 0.000 | 优先缩框，去掉过多背景，统一边界风格。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0347_oversized_box_sceneadd1100521_jpg.rf.e87ffc7adc2925a7cc54323f23a52366.jpg |
| 19 | B | sceneadd700441_jpg.rf.aefa9498d5ee520876e1a066876093c9.jpg | 2 | scratch | missing_label | 0.612 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0092_missing_label_sceneadd700441_jpg.rf.aefa9498d5ee520876e1a066876093c9.jpg |
| 20 | B | sceneadd2104341_jpg.rf.12b789058677b1b15599a011d0156241.jpg | 2 | rust | missing_label | 0.596 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0102_missing_label_sceneadd2104341_jpg.rf.12b789058677b1b15599a011d0156241.jpg |
| 21 | B | image-020-3-1-_jpg.rf.808fb1e2b472ae60d6e0c6e6d78884bc.jpg | 2 | missing-rivet | missing_label | 0.567 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0128_missing_label_image-020-3-1-_jpg.rf.808fb1e2b472ae60d6e0c6e6d78884bc.jpg |
| 22 | B | image-034_jpg.rf.c66a07c6c2fc50f5a5f104a32122e972.jpg | 2 | scratch | missing_label | 0.530 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0159_missing_label_image-034_jpg.rf.c66a07c6c2fc50f5a5f104a32122e972.jpg |
| 23 | B | sceneadd302001_jpg.rf.6ba0a4a827ee3589e974fd6fa433cb50.jpg | 2 | scratch | missing_label | 0.818 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0026_missing_label_sceneadd302001_jpg.rf.6ba0a4a827ee3589e974fd6fa433cb50.jpg |
| 24 | B | sceneadd1000261_jpg.rf.c7bc19aaddd28b8bcc43ed34d89dc1b1.jpg | 2 | rust | missing_label | 0.591 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0132_missing_label_sceneadd1000261_jpg.rf.c7bc19aaddd28b8bcc43ed34d89dc1b1.jpg |
| 25 | B | sceneadd701621_jpg.rf.c8afe2a4210b42a6623ecec9b7ad75d9.jpg | 2 | missing-rivet | missing_label | 0.757 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0025_missing_label_sceneadd701621_jpg.rf.c8afe2a4210b42a6623ecec9b7ad75d9.jpg |
| 26 | C | sceneadd1000561_jpg.rf.b57090ad28bb75612f13d6455f5c1e79.jpg | 2 | scratch | missing_label | 0.686 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0153_missing_label_sceneadd1000561_jpg.rf.b57090ad28bb75612f13d6455f5c1e79.jpg |
| 27 | C | image-043_jpg.rf.462cc2e2ad3a4089a62c1c0d690c398b.jpg | 2 | missing-rivet | wrong_class | 0.445 | 优先核查类别定义，必要时改类而不是重画框。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0199_wrong_class_image-043_jpg.rf.462cc2e2ad3a4089a62c1c0d690c398b.jpg |
| 28 | C | image-052-1-_jpg.rf.4a76628f0951fe5346ead442ec0645b5.jpg | 2 | scratch | missing_label | 0.706 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0023_missing_label_image-052-1-_jpg.rf.4a76628f0951fe5346ead442ec0645b5.jpg |
| 29 | C | image-048_jpg.rf.cba1f3bc2ce15c985cd345242baf6ef6.jpg | 2 | missing-rivet | missing_label | 0.670 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0047_missing_label_image-048_jpg.rf.cba1f3bc2ce15c985cd345242baf6ef6.jpg |
| 30 | C | image-020_jpg.rf.bd1740f1031f7e4ddd898c667abcae81.jpg | 2 | scratch | missing_label | 0.637 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0070_missing_label_image-020_jpg.rf.bd1740f1031f7e4ddd898c667abcae81.jpg |
| 31 | C | image-054-1-_jpg.rf.275c89d186fae1eb7b0eb0f1dd49091a.jpg | 2 | scratch | missing_label | 0.569 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0125_missing_label_image-054-1-_jpg.rf.275c89d186fae1eb7b0eb0f1dd49091a.jpg |
| 32 | C | image-048-1-_jpg.rf.5850e0c3b8b284214209280affcc5b4d.jpg | 2 | rust | missing_label | 0.559 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0133_missing_label_image-048-1-_jpg.rf.5850e0c3b8b284214209280affcc5b4d.jpg |
| 33 | C | image-063_jpg.rf.a92c843890b20a788c55a34a29ff7801.jpg | 2 | scratch | missing_label | 0.555 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0137_missing_label_image-063_jpg.rf.a92c843890b20a788c55a34a29ff7801.jpg |
| 34 | C | image-070-1-_jpg.rf.560f3e3d7c1a8ff9c0e179b064b8befa.jpg | 2 | scratch | missing_label | 0.551 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0140_missing_label_image-070-1-_jpg.rf.560f3e3d7c1a8ff9c0e179b064b8befa.jpg |
| 35 | C | image-018-1-_jpg.rf.d7b3d6897e4e6264951fc907538e3dcf.jpg | 2 | rust | missing_label | 0.532 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0156_missing_label_image-018-1-_jpg.rf.d7b3d6897e4e6264951fc907538e3dcf.jpg |
| 36 | C | scene1000861_jpg.rf.5b999e144e8a99b6b72c11eb4cf4889a.jpg | 2 | rust | missing_label | 0.514 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0176_missing_label_scene1000861_jpg.rf.5b999e144e8a99b6b72c11eb4cf4889a.jpg |
| 37 | C | image-065_jpg.rf.922257691c843773f686313f9459761d.jpg | 2 | scratch | missing_label | 0.513 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0177_missing_label_image-065_jpg.rf.922257691c843773f686313f9459761d.jpg |
| 38 | C | sceneadd1300541_jpg.rf.bd726c7308fc05b2e761a633f7f44fb5.jpg | 2 | rust | missing_label | 0.509 | 优先核查是否漏标，若确认是真缺陷则补框并补类别。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0180_missing_label_sceneadd1300541_jpg.rf.bd726c7308fc05b2e761a633f7f44fb5.jpg |
| 39 | C | sceneadd1301261_jpg.rf.494baa1f16efbeec40d3010d18864ed4.jpg | 2 | rust | wrong_class | 0.727 | 优先核查类别定义，必要时改类而不是重画框。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0201_wrong_class_sceneadd1301261_jpg.rf.494baa1f16efbeec40d3010d18864ed4.jpg |
| 40 | C | sceneadd1401041_jpg.rf.bebdb52f9929a33a5942823e453fea69.jpg | 2 | scratch | wrong_class | 0.571 | 优先核查类别定义，必要时改类而不是重画框。 | D:\桌面\毕设\数据集\yolo-baseline-repro\analysis\round14_annotation_audit\vis\0196_wrong_class_sceneadd1401041_jpg.rf.bebdb52f9929a33a5942823e453fea69.jpg |

## Tier Guide

- `A` 级：前 10 张，优先立刻修。这一批最可能影响 clean_train 首轮收益。
- `B` 级：第 11 到 25 张，建议在 `A` 级完成后立即跟进。
- `C` 级：其余图，作为第二轮补修清单。

## Outputs

- Image priority CSV: `analysis\round14_annotation_audit\R14_QA2_clean_train_image_priority.csv`
- Class priority CSV: `analysis\round14_annotation_audit\R14_QA2_clean_train_class_priority.csv`
- Label fix log template: `analysis\round14_annotation_audit\label_fix_log_template.csv`