# 失败与未验证项

原始CPU/下载日志只读保存于/home/lzf/ldx/tmp/cpgc_sep_20261028/，Google Drive服务器直连超时后，使用Windows网络→SSH标准输入流下载到服务器独立资源目录，每个文件独立SHA与receipt，未覆盖旧资源。RaSa ICFG/RSTP、UAP、ALBEF及SEP资源均已准备。

SEP人工全并列Fixture触发官方NaN，此为CPU边界证据，不是GPU失败，不自动修改官方算法。真实语义等价性尚未验证。

正式实验任何FAIL以对应Attempt/FAIL.json及控制日志为准。尚未完成的结果保留空值，不能用论文/Smoke数值填充；无真实完成证据前不得宣称12组完成。

最终启动门禁RESOURCE_BUSY：GPU仅1302MiB空闲。此为Attempt创建前资源阻塞，不是GPU算法FAIL。无GPU/Full/Watcher进程，无数值失败Attempt。详见RESOURCE_STATUS.json。
