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


DEVICE=:0 env PYTORCH_JIT=0 python entity_main.py --algo dreamer --env simple_white --action-repeat 2 --id simple_white_debug_mlr1e-4_alr1e-5_vlr1e-5_mel50_se200 --model_learning-rate 1e-4 --actor_learning-rate 1e-5 --value_learning-rate 1e-5 --max-episode-length 50 --seed-episodes 200

env PYTORCH_JIT=0 CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python main.py --algo dreamer --env walker-walk --action-repeat 2 --id debug_4_9_2021


DEVICE=:0 env PYTORCH_JIT=0 python entity_main.py --algo dreamer --env simple_white --action-repeat 2 --id simple_white_debug_mlr1e-4_alr1e-5_vlr1e-5_mel10_se1000_e1000000_fix_rendering_scale_action --model_learning-rate 1e-4 --actor_learning-rate 1e-5 --value_learning-rate 1e-5 --max-episode-length 10 --seed-episodes 1000 --episodes 1000000

4/13/21
DEVICE=:0 env PYTORCH_JIT=0 python entity_main.py --algo dreamer --env simple_white --action-repeat 2 --id simple_white_debug_mlr1e-4_alr1e-5_vlr1e-5_mel100_se1000_e1000000_fix_rendering_scale_action --model_learning-rate 1e-4 --actor_learning-rate 1e-5 --value_learning-rate 1e-5 --max-episode-length 100 --seed-episodes 1000 --episodes 1000000

CUDA_VISIBLE_DEVICES=4 DEVICE=:0 env PYTORCH_JIT=0 python entity_main.py --algo dreamer --env simple_white --action-repeat 2 --id simple_white_debug_mlr1e-4_alr1e-5_vlr1e-5_mel100_se1000_e1000000_fix_rendering_scale_action_big --model_learning-rate 1e-4 --actor_learning-rate 1e-5 --value_learning-rate 1e-5 --max-episode-length 100 --seed-episodes 1000 --episodes 1000000

CUDA_VISIBLE_DEVICES=7 DEVICE=:0 env PYTORCH_JIT=0 python entity_main.py --algo dreamer --env simple_box --action-repeat 2 --id simple_box_debug_mlr1e-4_alr1e-5_vlr1e-5_mel100_se1000_e1000000_fix_rendering_scale_action_big --model_learning-rate 1e-4 --actor_learning-rate 1e-5 --value_learning-rate 1e-5 --max-episode-length 100 --seed-episodes 1000 --episodes 1000000

4/14/21
Debugging modular
env PYTORCH_JIT=0 python modular_main.py --algo dreamer --env walker-walk --action-repeat 2 --id debug0 --episodes 10 --batch-size 8 --chunk-size 20 --collect-interval 2 --global-kl-beta 1e-1 --overshooting-kl-beta 1e-1 --overshooting-reward-scale 1e-1 --learning-rate-schedule 10 --max-episode-length 20 --test-interval 1

4/15/21
k=1, object-centric dreamer (dreamer5)
CUDA_VISIBLE_DEVICES=2 DEVICE=:0 env PYTORCH_JIT=0 python modular_main.py --algo dreamer --env simple_box --action-repeat 2 --id simple_box_debug_mlr1e-4_alr1e-5_vlr1e-5_mel100_se1000_e1000000_fix_rendering_scale_action_big --model_learning-rate 1e-4 --actor_learning-rate 1e-5 --value_learning-rate 1e-5 --max-episode-length 100 --seed-episodes 1000 --episodes 1000000 --slots --num_slots 1 --batch-size 10

(dreamer4)
CUDA_VISIBLE_DEVICES=4 DEVICE=:0 env PYTORCH_JIT=0 python modular_main.py --algo dreamer --env simple_box4 --action-repeat 2 --id simple_box4_debug_mlr1e-4_alr1e-5_vlr1e-5_mel100_se1000_e1000000_fix_rendering_scale_action_big --model_learning-rate 1e-4 --actor_learning-rate 1e-5 --value_learning-rate 1e-5 --max-episode-length 100 --seed-episodes 1000 --episodes 1000000 --slots --num_slots 5 --batch-size 10