
MODEL_PATH="classifier/meta/2202180921/model/resnet50.pt"

for cpt in 1 6 7
do
  for cmp in mozart scriabin debussy scarlatti liszt schubert chopin bach brahms haydn beethoven schumann rachmaninoff
  do
    python -m supervised.test_tcav_significance $MODEL_PATH --concept $cpt --omit-onset --composer $cmp --save-dir sign_${cmp}_${cpt} --layers layer4
  done
done