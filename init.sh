# download necessary dependencies
pip install -r requirements.txt

# make imgs directory if it doesn't exist
mkdir -p ./imgs

# get the tar.gz file from the cifar10 website
http GET https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz >> cifar100 
tar -xf cifar100 --directory ./imgs
if test -d "./imgs/cifar-100-python"
then
    echo "Cifar100 file downloaded!"
    rm cifar100
fi

# creates env file that will be used to start the api process.
echo "PORT=5000" >> .env
if test -f ".env"
then
    echo "Environment File Created. Make sure it contains 'PORT=5000'"
fi


