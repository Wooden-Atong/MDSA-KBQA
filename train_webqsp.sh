# basic NSM model
CUDA_VISIBLE_DEVICES=0 python main_nsm.py --model_name gnn --data_folder data/webqsp/ --checkpoint_dir checkpoint/webqsp_pretrain/ --batch_size 20 --test_batch_size 40 --num_step 3 --entity_dim 50 --word_dim 300 --kg_dim 100 --kge_dim 100 --eval_every 2 --encode_type --experiment_name webqsp_nsm --eps 0.95 --num_epoch 200 --use_self_loop --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb

# hybrid teacher student
CUDA_VISIBLE_DEVICES=0 python main_teacher.py --model_name gnn --teacher_type hybrid --data_folder data/webqsp/ --checkpoint_dir checkpoint_new/webqsp_teacher/ --batch_size 20 --test_batch_size 40 --num_step 3 --entity_dim 50 --word_dim 300 --kg_dim 100 --kge_dim 100 --eval_every 2 --experiment_name webqsp_hybrid_teacher --eps 0.95 --num_epoch 100 --use_self_loop --lr 5e-4 --q_type seq --word_emb_file word_emb_300d.npy --encode_type --loss_type kl --constrain_type js --reason_kb --load_experiment ../webqsp_pretrain/webqsp_nsm-final.ckpt --lambda_constrain 0.01 --lambda_back 0.1
CUDA_VISIBLE_DEVICES=0 python main_student.py --model_name gnn --teacher_model gnn --teacher_type hybrid --data_folder data/webqsp/ --checkpoint_dir checkpoint/webqsp_student/ --batch_size 20 --test_batch_size 40 --num_step 3 --entity_dim 50 --word_dim 300 --kg_dim 100 --kge_dim 100 --eval_every 2 --experiment_name webqsp_hybrid_student --eps 0.95 --num_epoch 180 --use_self_loop --lr 5e-4 --q_type seq --word_emb_file word_emb_300d.npy --encode_type --loss_type kl --constrain_type js --reason_kb --load_teacher ../webqsp_teacher/webqsp_hybrid_teacher-final.ckpt --lambda_label 0.05

# # parallel teacher student
CUDA_VISIBLE_DEVICES=0 python main_teacher.py --model_name gnn --teacher_type parallel --data_folder data/webqsp/ --checkpoint_dir checkpoint/webqsp_teacher/ --batch_size 20 --test_batch_size 40 --num_step 3 --entity_dim 50 --word_dim 300 --kg_dim 100 --kge_dim 100 --eval_every 2 --experiment_name webqsp_parallel_teacher --eps 0.95 --num_epoch 100 --use_self_loop --lr 5e-4 --q_type seq --word_emb_file word_emb_300d.npy --encode_type --loss_type kl --reason_kb --constrain_type js --lambda_constrain 0.01 --lambda_back 0.1 --load_pretrain ../webqsp_pretrain/webqsp_nsm-final.ckpt
CUDA_VISIBLE_DEVICES=0 python main_student.py --model_name gnn --teacher_model gnn --teacher_type parallel --data_folder data/webqsp/ --checkpoint_dir checkpoint/webqsp_student/ --batch_size 20 --test_batch_size 40 --num_step 3 --entity_dim 50 --word_dim 300 --kg_dim 100 --kge_dim 100 --eval_every 2 --experiment_name webqsp_parallel_student --eps 0.95 --num_epoch 180 --use_self_loop --lr 5e-4 --q_type seq --word_emb_file word_emb_300d.npy --encode_type --loss_type kl --constrain_type js --reason_kb --load_teacher ../webqsp_teacher/webqsp_parallel_teacher-final.ckpt --lambda_label 0.05





