if [ -z "$1" ]
then
 echo "You cannot do 'git push' without a proper commit message!"
else
 git add .
 git commit -m "$1"
 git push -u origin main
fi
