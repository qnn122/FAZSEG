# Model
echo '---------- Setting up Model --------------'
gdown 'https://drive.google.com/uc?id=1SxJQeT37bgdJgvZHAqrhNnPRBbaqT_Ax'
mkdir models # TODO: add check models/ exists
mv 'Se_resnext50-920eef84.pth' models/
echo 'Done'

# Data
echo '---------- Setting up Data --------------'
gdown 'https://drive.google.com/uc?id=1vr5MOdmyHYSxx4QugDZG-esASQ5A_THu'
mkdir data # TODO: add check models/ exists
unzip data.zip -d data/
rm data.zip
echo 'Done'