# orig
env PYTORCH_JIT=0 python main.py --algo dreamer --env walker-walk --action-repeat 2 --id debug0 --episodes 10 --batch-size 8 --chunk-size 20 --collect-interval 4 --global-kl-beta 1e-1 --overshooting-kl-beta 1e-1 --overshooting-reward-scale 1e-1 --learning-rate-schedule 10

# slots
env PYTORCH_JIT=0 python main.py --algo dreamer --env walker-walk --action-repeat 2 --id debug0 --episodes 10 --batch-size 8 --chunk-size 20 --collect-interval 4 --global-kl-beta 1e-1 --overshooting-kl-beta 1e-1 --overshooting-reward-scale 1e-1 --learning-rate-schedule 10 --slots

# slots GPU (batch size 10)
CUDA_VISIBLE_DEVICES=5 DISPLAY=:0 python main.py --algo dreamer --env walker-walk --action-repeat 2 --slots --id slots0 --batch-size 10


# entity
env PYTORCH_JIT=0 python entity_main.py --algo dreamer --env simple --action-repeat 2

4/11/21
env PYTORCH_JIT=0 python entity_main.py --algo dreamer --env simple_white --action-repeat 2 --id simple_white_debug --collect-interval 1 --test-interval 1


4/12/21
works on alan
DEVICE=:0 env PYTORCH_JIT=0 python entity_main.py --algo dreamer --env simple_white --action-repeat 2 --id simple_white_debug --collect-interval 1 --test-interval 1

use 

os.environ["SDL_VIDEODRIVER"] = "dummy"

instead of xvfb in order to make it work with pygame. WOrks with CUDA too