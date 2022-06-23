#!/bin/bash
rm experiments.db
python ../quesiton-gen/question_gen/gen_questions.py init-db experiments.db schema.sql
python ../quesiton-gen/question_gen/gen_questions.py gen-random-questions experiments.db miner 10 traj