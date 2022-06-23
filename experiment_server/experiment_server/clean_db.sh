#!/bin/bash
rm experiments.db
python gen_questions.py init-db experiments.db schema.sql
python gen_questions.py gen-random-questions experiments.db miner 5 traj