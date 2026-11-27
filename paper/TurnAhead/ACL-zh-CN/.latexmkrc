# 中文稿与模板统一使用 XeLaTeX，并生成源文件与 PDF 的同步信息。
$pdf_mode = 5;
$xelatex = 'xelatex -synctex=1 -interaction=nonstopmode -halt-on-error -file-line-error %O %S';
