# TurnAhead 中文阅读版

- `main.pdf`：TurnAhead 论文中文稿，对应 `main.tex` 和 `sections/`。
- `acl_latex.pdf`：ACL 模板说明中文稿，对应 `acl_latex.tex`。

正文、标题、图示、算法、定义、命题及附录证明均已翻译。英文原稿中的未完成内容继续标为“【计划补充】”，未补写实验结果。公式、数值、引用键及标签保持对应关系。

参考文献标题使用中文译名，英文原题保存在 `.bib` 文件的 `origtitle` 字段中；作者、出版物名称、链接等原始出版信息保留。译名仅为阅读辅助。`acl.sty`、`acl_natbib.bst` 为排版程序，保持原样；显示文字的中文设置集中在 `chinese.tex`。

## 重新编译

在本目录运行：

```bash
latexmk -xelatex main.tex
latexmk -xelatex acl_latex.tex
```

本目录的 `.latexmkrc` 已将默认引擎设为 XeLaTeX，因此也可直接运行 `latexmk main.tex`。两份文档均生成 SyncTeX 信息，方便在源码与 PDF 间跳转。

当前环境已具备 XeLaTeX、BibTeX、latexmk、xeCJK，以及 Noto CJK、Nimbus、DejaVu 字体。中文字体与拉丁字体配置见 `chinese.tex`。

中文稿保留 ACL 双栏、审稿行号和页码。英文原稿位于相邻的 `../ACL/` 目录。
