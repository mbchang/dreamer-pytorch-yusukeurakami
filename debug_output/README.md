# orig
env PYTORCH_JIT=0 python main.py --algo dreamer --env walker-walk --action-repeat 2 --id debug0 --episodes 10 --batch-size 8 --chunk-size 20 --collect-interval 4 --global-kl-beta 1e-1 --overshooting-kl-beta 1e-1 --overshooting-reward-scale 1e-1 --learning-rate-schedule 10

# slots
env PYTORCH_JIT=0 python main.py --algo dreamer --env walker-walk --action-repeat 2 --id debug0 --episodes 10 --batch-size 8 --chunk-size 20 --collect-interval 4 --global-kl-beta 1e-1 --overshooting-kl-beta 1e-1 --overshooting-reward-scale 1e-1 --learning-rate-schedule 10 --slots