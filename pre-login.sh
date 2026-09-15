#!/bin/bash
sudo mkdir -p -m 755 /etc/apt/keyrings 	&& out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg && cat $out | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null && sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg && sudo mkdir -p -m 755 /etc/apt/sources.list.d && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null 	&& sudo apt update 	&& sudo apt install gh -y
echo "Run gh auth login"

# To disconnect without closing your processes, press ctrl+b, release, then d. 
# To disable auto-tmux, run `touch ~/.no_auto_tmux` and reconnect. See also https://tmuxcheatsheet.com/