# TechQA：104 份最终回答的来源复核

这是正式自动评分之外的探索性分析。判断针对给定历史文档的关键规则，不验证回答中的全部额外生产环境建议，也不把 F1 当作语义正确率。它不是独立盲审：第一题未隐藏方法名；之后按答案去重、用散列 ID 隐藏方法名展示，但复核者已见过部分早期输出。方法暴露与主观归类可能产生偏差。未使用这些标签调整模型、检索器或触发阈值。

“关键结论符合”只表示所问的核心修复/规则得到来源支持；“部分匹配”包括遗漏关键位置、替代命令未被来源验证、截断或混有不可靠额外内容。明确与核心规则相反或没有有效技术内容的回答记为“与参考不符”。空白单独列出。所有标签与原始轨迹 SHA256 关联。

完整问题、gold 和全部实际引用/gold 文档见压缩包内 data/techqa；完整回答、推理、token 与注入事件见 snapshot/techqa.jsonl。本文件的文档 ID 命中仅按标注计算，不是对所有替代证据的语义评判。

## DEV_Q209 — How to change the maximum string length for properties in FileNet Administration Console for Content Engine (ACCE)

ACCE 的 Class Definition → Properties → Property Definitions 下拉选择属性，在单独标签页修改，并回到类定义保存。SC 下降方法检索到替代文档 swg21680952，包含关键路径，但没有命中唯一 gold 文档 ID；标注 ID 不是全部可用证据的清单。

标注文档：`swg21696083.txt`。

| 方法 | 回答 F1 | 注入偏移 / 标注 ID 命中 | 来源复核 | 说明 |
|---|---:|---|---|---|
| closed-book | 22.7% | 未注入 | 与参考不符 | 称 ACCE GUI 无法修改，转向 SDK；与源文档的 ACCE 操作步骤相反。 |
| static-user-bm25 | 33.2% | 0 / 否 | 部分匹配 | 提到编辑 Property Definition，但给出不可靠的导航路径，未说明 Property Definitions 下拉及新标签页，回答被截断。 |
| start-bm25 | 27.1% | 0 / 否 | 与参考不符 | 明确说只能在 FEM 而非 ACCE 修改。 |
| fixed32-bm25 | 27.1% | 32 / 否 | 部分匹配 | 指向 Property Definition，避免 Property Template；缺少问题所需的下拉框/独立标签页位置，增加了传播与超时细节。 |
| fixed128-bm25 | 20.5% | 128 / 否 | 与参考不符 | 延续 ACCE GUI 不支持的错误前提，并补出 SDK/脚本路径。 |
| lowprob-bm25 | 16.3% | 256 / 否 | 与参考不符 | 检索到相似但不同的 Saved Definitions 5MB 限制，把它用于当前属性长度问题。 |
| rule-bm25 | 22.7% | 未注入 | 与参考不符 | 未触发，与闭卷逐 token 相同，保留 ACCE GUI 不支持的错误。 |
| hybrid-trigger-bm25 | 16.3% | 256 / 否 | 与参考不符 | 同低概率触发，在 256-token 预算边界检索，转向无关 5MB 限制。 |
| sentence-bm25 | 31.7% | 56 / 否 | 与参考不符 | 把应编辑的 Property Definition 改为 Property Template，与支持材料也不符。 |
| preanswer-bm25 | 0.0% | 256 / 否 | 空回答 | 推理结束后资料已注入，随后只生成 EOS，没有最终回答。 |
| fixed32-char | 32.2% | 32 / 否 | 与参考不符 | 给出 Property Template / Content tab 路径，并增加来源未支持的 cmtmedi 操作。 |
| fixed32-hybrid | 27.1% | 32 / 否 | 部分匹配 | 与固定32 BM25 的文档顺序及最终轨迹相同。 |
| scdrop-bm25 | 36.5% | 64 / 否 | 关键结论符合 | 给出 Class Definition → Properties → Property Definitions → 独立标签页的关键路径；检索命中替代支持文档 swg21680952，非 gold 标注文档。 |

## DEV_Q195 — TLS protocol with ITCAM for Datapower

同一 HotFix 支持 TLSv1.2。若 appliance 禁用 TLSv1，需手动设置 ITCAM 的 KBN_SOMA_PROTOCOL=TLSv1.2；只调整 appliance Crypto Profile 未回答客户端配置问题。

标注文档：`swg21959224.txt`。

| 方法 | 回答 F1 | 注入偏移 / 标注 ID 命中 | 来源复核 | 说明 |
|---|---:|---|---|---|
| closed-book | 2.3% | 未注入 | 与参考不符 | 否定 TLS1.2 支持并要求升级，与参考的同一 HotFix 可支持 TLS1.2 相反。 |
| static-user-bm25 | 3.7% | 0 / 否 | 与参考不符 | 给出 appliance 端配置，未处理 ITCAM 7.1 的协议配置。 |
| start-bm25 | 8.0% | 0 / 否 | 与参考不符 | 声称 ITCAM protocol 设置不能指定版本；与参考手动设置 KBN_SOMA_PROTOCOL 的修复相反。 |
| fixed32-bm25 | 4.1% | 32 / 否 | 与参考不符 | 否定 ITCAM 可选 TLS1.2，转到 appliance Crypto Profile；未给出参考中的 HotFix 与 KBN_SOMA_PROTOCOL。 |
| fixed128-bm25 | 5.4% | 128 / 否 | 与参考不符 | 仅设置 appliance Crypto Profile，未说明 ITCAM 的 KBN_SOMA_PROTOCOL 覆盖。 |
| lowprob-bm25 | 2.3% | 未注入 | 与参考不符 | 否定 TLS1.2 支持并要求升级，与参考的同一 HotFix 可支持 TLS1.2 相反。 |
| rule-bm25 | 0.0% | 160 / 否 | 空回答 | 最终回答只有空白。 |
| hybrid-trigger-bm25 | 0.0% | 160 / 否 | 空回答 | 最终回答只有空白。 |
| sentence-bm25 | 7.3% | 37 / 否 | 与参考不符 | 仅调整 DataPower appliance profile，未解决 ITCAM 客户端默认 TLSv1 的配置。 |
| preanswer-bm25 | 0.0% | 256 / 否 | 与参考不符 | 只输出引用结束标记 [End reference]，没有技术回答。 |
| fixed32-char | 15.0% | 32 / 是 | 关键结论符合 | 包含同一 HotFix 支持 TLS1.2 及 KBN_SOMA_PROTOCOL=TLSv1.2 的核心修复。 |
| fixed32-hybrid | 7.3% | 32 / 是 | 关键结论符合 | 明确给出 KBN_SOMA_PROTOCOL=TLSv1.2 和默认 TLSv1.0 的关键原因；额外的 appliance 配置不作为正确性的依据。 |
| scdrop-bm25 | 2.3% | 未注入 | 与参考不符 | 否定 TLS1.2 支持并要求升级，与参考的同一 HotFix 可支持 TLS1.2 相反。 |

## DEV_Q239 — Cannot launch RFT 8.2

清理用户 .eclipse/ibm.software.delivery.platform_7.5.0_???? 旧配置目录并重新启动；来源没有把该问题诊断为 JDK7 要求或目录权限。

标注文档：`swg21330011.txt`。

| 方法 | 回答 F1 | 注入偏移 / 标注 ID 命中 | 来源复核 | 说明 |
|---|---:|---|---|---|
| closed-book | 0.0% | 未注入 | 与参考不符 | 改 JDK 7/重装而未清理源文档指出的 .eclipse 配置目录；JDK7要求不在参考中。 |
| static-user-bm25 | 22.0% | 0 / 否 | 部分匹配 | 识别旧 .eclipse 配置并建议清理重启，但范围扩为整个 .eclipse，未定位 ibm.software.delivery.platform_7.5.0_???? 子目录；另加权限建议。 |
| start-bm25 | 17.3% | 0 / 否 | 部分匹配 | 识别用户配置缓存过期，但给出未被来源支持的 com.ibm.rational.rft_8.2.0 目录，而非参考的平台目录。 |
| fixed32-bm25 | 8.6% | 32 / 否 | 与参考不符 | 清理工作区 .metadata 与源文档要求的用户 .eclipse 平台目录不同，未给出参考修复。 |
| fixed128-bm25 | 4.2% | 128 / 否 | 与参考不符 | 以 JDK7/重装为主要修复，未给出参考的用户 Eclipse 配置清理。 |
| lowprob-bm25 | 0.0% | 未注入 | 与参考不符 | 改 JDK 7/重装而未清理源文档指出的 .eclipse 配置目录；JDK7要求不在参考中。 |
| rule-bm25 | 4.2% | 128 / 否 | 与参考不符 | 以 JDK7/重装为主要修复，未给出参考的用户 Eclipse 配置清理。 |
| hybrid-trigger-bm25 | 4.2% | 128 / 否 | 与参考不符 | 以 JDK7/重装为主要修复，未给出参考的用户 Eclipse 配置清理。 |
| sentence-bm25 | 7.0% | 43 / 否 | 与参考不符 | 明确把问题归为权限并排除缓存问题，与参考诊断不符。 |
| preanswer-bm25 | 0.0% | 256 / 否 | 空回答 | 最终回答为空。 |
| fixed32-char | 8.6% | 32 / 否 | 与参考不符 | 清理工作区 .metadata 与源文档要求的用户 .eclipse 平台目录不同，未给出参考修复。 |
| fixed32-hybrid | 8.6% | 32 / 否 | 与参考不符 | 清理工作区 .metadata 与源文档要求的用户 .eclipse 平台目录不同，未给出参考修复。 |
| scdrop-bm25 | 13.6% | 64 / 否 | 与参考不符 | 把原因定为 SDP/IMShared 权限；未给出源文档的用户 .eclipse 旧配置清理。 |

## DEV_Q160 — How to increase the HTTP Session time-out value for Workplace application on WebSphere Application Server?

位置与优先级从低到高为 server → Enterprise Application → Web Application → web.xml → runtime code；控制台子级需选择 Override。原 gold 只摘录了位置，未摘录完整优先级，因此 F1 不能直接判定回答是否覆盖问题。

标注文档：`swg21163875.txt`。

| 方法 | 回答 F1 | 注入偏移 / 标注 ID 命中 | 来源复核 | 说明 |
|---|---:|---|---|---|
| closed-book | 20.8% | 未注入 | 与参考不符 | 把 web.xml 当最高级，遗漏运行时代码，混合 module/EAR 层级并给出不受参考支持的配置路径。 |
| static-user-bm25 | 23.0% | 0 / 是 | 关键结论符合 | 完整给出五级位置及顺序，指出 web.xml 高于全部控制台配置、代码最高。 |
| start-bm25 | 18.8% | 0 / 是 | 关键结论符合 | 五级优先级（code > web.xml > Web应用 > 企业应用 > server）完整正确；末尾虽截断，核心表已给出。LTPA补充不计入核心判定。 |
| fixed32-bm25 | 18.5% | 32 / 是 | 关键结论符合 | 明确列出 code、web.xml、Web Module、企业应用、server 的完整优先级及位置。 |
| fixed128-bm25 | 24.2% | 128 / 是 | 部分匹配 | 给出控制台三级与 web.xml，但截断前未给出运行时代码的最高优先级，还残留引用标记。 |
| lowprob-bm25 | 20.8% | 未注入 | 与参考不符 | 把 web.xml 当最高级，遗漏运行时代码，混合 module/EAR 层级并给出不受参考支持的配置路径。 |
| rule-bm25 | 18.3% | 96 / 是 | 部分匹配 | 声称只有四个级别，优先级表遗漏控制台 Web Application 一级；后文提到该配置位置但未补全五级顺序。 |
| hybrid-trigger-bm25 | 18.3% | 96 / 是 | 部分匹配 | 声称只有四个级别，优先级表遗漏控制台 Web Application 一级；后文提到该配置位置但未补全五级顺序。 |
| sentence-bm25 | 19.8% | 40 / 是 | 关键结论符合 | 五级配置与覆盖规则完整，符合源文档。 |
| preanswer-bm25 | 0.0% | 256 / 是 | 与参考不符 | 仅输出省略号，没有技术回答。 |
| fixed32-char | 18.5% | 32 / 是 | 关键结论符合 | 明确列出 code、web.xml、Web Module、企业应用、server 的完整优先级及位置。 |
| fixed32-hybrid | 18.5% | 32 / 是 | 关键结论符合 | 明确列出 code、web.xml、Web Module、企业应用、server 的完整优先级及位置。 |
| scdrop-bm25 | 17.1% | 64 / 是 | 关键结论符合 | 完整列出五个位置、优先级与控制台 Override 条件。 |

## DEV_Q011 — How to proceed when jextract utility is throwing an OutOfMemory error ?

增大 jextract 自身的 JVM 堆，明确支持的命令参数是 -J-Xmx。来源同时说明工具需与生成 core 的系统/安装匹配。实际 token 解码核对显示，本题全部注入片段均未包含 -J-Xmx；虽命中文档，前部截取没有暴露该关键参数。解码记录和 tokenizer SHA256 见归档 snapshot/evidence-exposure*。

标注文档：`swg21577379.txt`。

| 方法 | 回答 F1 | 注入偏移 / 标注 ID 命中 | 来源复核 | 说明 |
|---|---:|---|---|---|
| closed-book | 20.9% | 未注入 | 部分匹配 | 正确识别工具自身的堆不足；只给环境变量调用，缺少来源明确的 -J-Xmx。 |
| static-user-bm25 | 22.8% | 0 / 是 | 部分匹配 | 识别应增加 jextract 堆，但首个 java -Xmx .../jextract 调用不受来源支持；未给明确的 -J-Xmx。 |
| start-bm25 | 9.6% | 0 / 是 | 部分匹配 | 包含正确的 jextract -J-Xmx 修复，但伪造来源续写和文档ID，并把压缩引用的32/64位规则说反；不能视为整体可靠回答。 |
| fixed32-bm25 | 26.7% | 32 / 是 | 部分匹配 | 建议增加工具堆，但未给文档的 -J-Xmx；JAVA_TOOL_OPTIONS 替代方式未在给定材料中验证。 |
| fixed128-bm25 | 15.9% | 128 / 是 | 部分匹配 | 诊断堆不足正确，但给出未经来源支持的 jextract.jar/JAVAOPTS 调用，没有 -J-Xmx。 |
| lowprob-bm25 | 20.9% | 未注入 | 部分匹配 | 正确识别工具自身的堆不足；只给环境变量调用，缺少来源明确的 -J-Xmx。 |
| rule-bm25 | 20.9% | 未注入 | 部分匹配 | 正确识别工具自身的堆不足；只给环境变量调用，缺少来源明确的 -J-Xmx。 |
| hybrid-trigger-bm25 | 20.9% | 未注入 | 部分匹配 | 正确识别工具自身的堆不足；只给环境变量调用，缺少来源明确的 -J-Xmx。 |
| sentence-bm25 | 13.1% | 58 / 是 | 与参考不符 | 转为提交原始内核转储或要求补丁，未给出直接增加 jextract 堆的修复。 |
| preanswer-bm25 | 15.2% | 256 / 是 | 部分匹配 | 识别 jextract 的 JVM 堆不足，但只给环境变量方案且回答截断；未给来源明确支持的 -J-Xmx 参数。 |
| fixed32-char | 10.6% | 32 / 是 | 与参考不符 | 明确否认 JVM 堆限制，把问题归为 OS 地址空间/swap，并建议修改系统参数；与参考的增大 jextract 堆相反。 |
| fixed32-hybrid | 13.3% | 32 / 是 | 与参考不符 | 把主要原因归为虚拟地址空间而非默认堆不足；仅条件性提到 -Xmx，没有来源的工具参数。 |
| scdrop-bm25 | 16.0% | 64 / 是 | 部分匹配 | 堆不足诊断正确，但把 jextract 路径当 Java 类调用，缺少 -J-Xmx，并给出未被来源支持的1.5倍规则。 |

## DEV_Q262 — Table ""."" could not be found

OS 与数据库均 UTF-8 时，CLP 不进行码页转换；若实际输入不是 Unicode/UTF-8，会保存错误码点并在读取时出错。题目的弯引号不能直接当作终止 SQL 字符串的 ASCII 单引号。

标注文档：`swg21412846.txt`。

| 方法 | 回答 F1 | 注入偏移 / 标注 ID 命中 | 来源复核 | 说明 |
|---|---:|---|---|---|
| closed-book | 12.3% | 未注入 | 与参考不符 | 把弯引号当作会终止 SQL 字符串的 ASCII 单引号，归因于转义而非来源中的错误编码字节。 |
| static-user-bm25 | 38.1% | 0 / 否 | 关键结论符合 | 正确指出 UTF-8 环境中写入非 UTF-8 评论数据导致错误码点。 |
| start-bm25 | 26.6% | 0 / 否 | 关键结论符合 | 核心编码错配原因正确；其对空表名字符串形成的细节属于额外解释。 |
| fixed32-bm25 | 28.1% | 32 / 否 | 关键结论符合 | 明确给出 OS/数据库均 UTF-8 而插入数据非 Unicode 导致错误码点的核心原因；具体 metadata/query 传播细节不作为参考已验证事实。 |
| fixed128-bm25 | 10.1% | 128 / 否 | 与参考不符 | 把问题改写成 ASCII 单引号未转义；没有识别 UTF-8 环境插入非 Unicode 数据的原因。 |
| lowprob-bm25 | 22.8% | 160 / 否 | 与参考不符 | 主要归因为未转义引号破坏查询，未说明双方均 UTF-8 时跳过转换导致错误字节；给出泛化配置建议。 |
| rule-bm25 | 12.3% | 未注入 | 与参考不符 | 把弯引号当作会终止 SQL 字符串的 ASCII 单引号，归因于转义而非来源中的错误编码字节。 |
| hybrid-trigger-bm25 | 22.8% | 160 / 否 | 与参考不符 | 主要归因为未转义引号破坏查询，未说明双方均 UTF-8 时跳过转换导致错误字节；给出泛化配置建议。 |
| sentence-bm25 | 32.4% | 38 / 否 | 关键结论符合 | 核心说明 OS/数据库 UTF-8 而评论数据包含非 Unicode 码点，符合参考；不据此验证其具体元数据解析路径。 |
| preanswer-bm25 | 31.3% | 256 / 否 | 部分匹配 | 后半段自我纠正为正确编码原因，但最终回答仍是未完成的推理和引用残留，前半段错误解析说法未清晰收束。 |
| fixed32-char | 26.4% | 32 / 否 | 关键结论符合 | 正确定位为 UTF-8 数据库中保存了旧代码页字节，应修正编码。 |
| fixed32-hybrid | 28.1% | 32 / 否 | 关键结论符合 | 明确给出 OS/数据库均 UTF-8 而插入数据非 Unicode 导致错误码点的核心原因；具体 metadata/query 传播细节不作为参考已验证事实。 |
| scdrop-bm25 | 27.4% | 96 / 否 | 关键结论符合 | 明确包含双方 UTF-8 而评论未按 Unicode 编码的核心原因；表名/查询细节不作为已验证内容。 |

## DEV_Q184 — Why am I seeing "CWSXM0100E: Multiple replies have been produced for a single request flow in module" ?

同一对 mediation primitives 之间存在视觉重叠的重复连线；移动节点检查，删除冗余线，保存并重新部署。

标注文档：`swg21584783.txt`。

| 方法 | 回答 F1 | 注入偏移 / 标注 ID 命中 | 来源复核 | 说明 |
|---|---:|---|---|---|
| closed-book | 9.9% | 未注入 | 与参考不符 | 归因于 Reply 变量冲突，未给出源文档的重复连线原因。 |
| static-user-bm25 | 19.4% | 0 / 否 | 与参考不符 | 归因于缺少 Reply/out terminal，未识别重复连线。 |
| start-bm25 | 11.7% | 0 / 否 | 与参考不符 | 归因于外部服务/代理重试和多份 SOAP 响应，而非图中重复连线。 |
| fixed32-bm25 | 29.7% | 32 / 是 | 关键结论符合 | 明确指出重叠的重复连线，并给出删除冗余线、保存、重新部署。 |
| fixed128-bm25 | 26.3% | 128 / 是 | 关键结论符合 | 正确识别重复连线，可移动节点暴露隐藏连线，删除后保存重部署。 |
| lowprob-bm25 | 9.9% | 未注入 | 与参考不符 | 归因于 Reply 变量冲突，未给出源文档的重复连线原因。 |
| rule-bm25 | 9.9% | 未注入 | 与参考不符 | 归因于 Reply 变量冲突，未给出源文档的重复连线原因。 |
| hybrid-trigger-bm25 | 9.9% | 未注入 | 与参考不符 | 归因于 Reply 变量冲突，未给出源文档的重复连线原因。 |
| sentence-bm25 | 29.7% | 32 / 是 | 关键结论符合 | 明确指出重叠的重复连线，并给出删除冗余线、保存、重新部署。 |
| preanswer-bm25 | 0.0% | 256 / 是 | 空回答 | 最终回答为空。 |
| fixed32-char | 34.7% | 32 / 是 | 关键结论符合 | 正确指出同一对 primitive 之间的隐藏重复线，并给出对应修复。 |
| fixed32-hybrid | 34.7% | 32 / 是 | 关键结论符合 | 正确指出同一对 primitive 之间的隐藏重复线，并给出对应修复。 |
| scdrop-bm25 | 9.9% | 未注入 | 与参考不符 | 归因于 Reply 变量冲突，未给出源文档的重复连线原因。 |

## DEV_Q249 — Help with Security Bulletin: IBM WebSphere MQ Administration command could cause denial of service (CVE-2016-8971)

IBM 公告 swg21998663。来源说明已认证且有 queue manager 权限的用户可能触发段错误和拒绝服务，CVSS 5.3；对应维护版本为 MQ 9.0.0.1 / 8.0.0.6。问句要求信息出处，只有准确摘要而无出处的回答记为部分匹配。

标注文档：`swg21998663.txt`。

| 方法 | 回答 F1 | 注入偏移 / 标注 ID 命中 | 来源复核 | 说明 |
|---|---:|---|---|---|
| closed-book | 15.0% | 未注入 | 与参考不符 | 编造公告ID2550623和不符合来源的修复版本，且回答残留推理并被截断。 |
| static-user-bm25 | 18.2% | 0 / 是 | 关键结论符合 | 准确给出公告 swg21998663 及关键影响和维护版本。 |
| start-bm25 | 50.7% | 0 / 是 | 关键结论符合 | 给出准确 IBM 公告 swg21998663、X-Force/CVE 入口及正确维护版本；末尾虽截断，出处和核心信息已完整。 |
| fixed32-bm25 | 36.3% | 32 / 是 | 关键结论符合 | 公告链接、认证权限、CVSS和修复版本正确。 |
| fixed128-bm25 | 9.2% | 128 / 是 | 关键结论符合 | 给出准确公告 URL 与两个修复版本，直接回答信息出处。 |
| lowprob-bm25 | 35.4% | 160 / 是 | 关键结论符合 | 准确给出官方公告位置、认证权限条件和版本信息。 |
| rule-bm25 | 15.0% | 未注入 | 与参考不符 | 编造公告ID2550623和不符合来源的修复版本，且回答残留推理并被截断。 |
| hybrid-trigger-bm25 | 35.4% | 160 / 是 | 关键结论符合 | 准确给出官方公告位置、认证权限条件和版本信息。 |
| sentence-bm25 | 17.5% | 35 / 是 | 部分匹配 | 技术概述和维护版本正确，但未给信息来源或公告链接。 |
| preanswer-bm25 | 18.4% | 256 / 是 | 关键结论符合 | 公告与修复版本正确；开头残留 </think> 标记，属于格式问题。 |
| fixed32-char | 36.7% | 32 / 是 | 关键结论符合 | 准确给出公告、权限条件、影响、CVSS和修复版本。 |
| fixed32-hybrid | 37.4% | 32 / 是 | 部分匹配 | 漏洞、CVSS、受影响和修复版本符合来源，但未给公告出处/编号或链接，未完整回答 where。 |
| scdrop-bm25 | 12.5% | 96 / 是 | 关键结论符合 | 公告编号/链接和受影响、修复版本均符合来源。 |
