# Bitrecs Validator Setup Guide

This guide ensures the Bitrecs validator works on **Ubuntu 24.10 LTS**. 

## 1. Installation script
Update your packages before running the install script.

**If you do not want to run the .sh script, start at [step 2](running_miner.md) of the miner guide to get python 3.11+venv setup first**

```bash
sudo apt-get update && sudo apt-get upgrade -y
curl -sL https://raw.githubusercontent.com/bitrecs/bitrecs-subnet/refs/heads/main/scripts/install_validator.sh | bash
```

## 2. Keys on machine and register
regen_coldkeypub

regen_hotkey

## 3. Environment Configuration

Before running the validator, edit the .env environment file and fill it in to match your config specs.

## 4. Firewall Configuration

**Warning:** port 22 is NOT required to be open for validators - we have it here to ensure you do not get disconnected if you activate UFW. The only required port is 7779 for validators. 

Configure the firewall using UFW:

```bash
sudo ufw allow 22
sudo ufw allow proto tcp to 0.0.0.0/0 port 7779
sudo ufw enable
sudo ufw reload
```

## 5. Start Validator (No Auto-Updates)
Monitor output with `pm2 logs`.
We recommend using --debug.trace for more verbosity.

```bash
pm2 start ./neurons/validator.py --name v -- \
        --netuid 122 \
        --wallet.name default --wallet.hotkey default \
        --neuron.vpermit_tao_limit 10_000 \
        --subtensor.network wss://entrypoint-finney.opentensor.ai:443 \
        --logging.trace \
        --r2.sync_on 

pm2 save
```

## 5.1 Start Validator (With Auto-Updates)

Use the start_validator.py script to run the auto-updater which will:

- create a pm2 process to handle updates automatically
- create a pm2 process to run the validator.py

see: [start_validator.py](/start_validator.py)

This is the recommended way for running a validator with worry free updates.