for file in ~/alvin/alvinHasLeftTheRoom/sounds10s/*
do
  python ~/alvin/alvinHasLeftTheRoom/main.py "$file" >> results10s.txt
done