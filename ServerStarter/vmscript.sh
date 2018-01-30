#!/usr/bin/env bash
pw="Venona99"
echo $pw | sudo -S apt-get install -y cmake zlib1g-dev xvfb libav-tools xorg-dev libboost-all-dev libsdl2-dev swig
sudo -H pip install gym[atari]
sudo -H pip install scikit-image
sudo sudo ufw allow 2222/tcp
sudo sudo ufw allow ssh
echo "y" | sudo -S ufw enable
