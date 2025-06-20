cn_template = lambda style_name, style_definition, text1, text2: \
f'''
你将看到两段文本。请根据下列定义和标准，判断哪一段文本的**语言风格（{style_name}）更强**，并说明理由。
---

**{style_name}强弱的定义：**
{style_definition}

---

**任务：**
请比较以下两段文本的{style_name}强弱：
* 文本A：{text1}
* 文本B：{text2}

你的任务是回答以下问题：
**哪一段文本的{style_name}更强？请说明理由。**

---

**输出格式：**
```
{style_name}更强的文本：[A / B]

理由：[请结合上面的{style_name}定义，进行简要说明。]
```
'''




en_template = lambda style_name, style_definition, text1, text2: \
f'''
You will see two pieces of text. Please evaluate which one demonstrates a **stronger writing style in terms of {style_name}**, based on the following definition and criteria.
---

**Definition of the strength of the writing style ({style_name}):**
{style_definition}

---

**Task:**
Please compare the following two texts in terms of the strength of their writing style ({style_name}):
* Text A: {text1}
* Text B: {text2}

Your task is to answer the following question:
**Which text demonstrates a stronger {style_name}? Please explain your reasoning.**

---

**Output format:**
```
Text with stronger {style_name}: [A / B]
Reason: [Briefly explain based on the definition of the {style_name} above.]

```
'''