# Release Update Instructions

## Miners
- make sure you are on the main branch
- pm2 stop your_pm2_id
- git pull
- pip install -e .
- pm2 restart your_pm2_id & pm2 logs


## Validators

### No Auto Updater
- make sure you are on the main branch
- pm2 stop your_pm2_id
- git pull
- pip install -e .
- pm2 restart your_pm2_id & pm2 logs


### Auto Updates Enabled
- Updates should be applied automatically
- check your logs to see if the update was applied correctly with no errors