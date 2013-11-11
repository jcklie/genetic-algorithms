all: show git

g: git

git:
	git add *
	git commit -v -m "autocommited by makefile"
	git push

u: update
	
update:
	git pull

edit:
	kate `find . | grep 'tex'` bibliography.bib
	
editall:
	kate `find . | grep 'tex'`

e: edit

s: show

sd:
	okular paper.pdf

show: pdf
	okular paper.pdf

pdf: 
	latexmk -bibtex -pdf paper.tex

