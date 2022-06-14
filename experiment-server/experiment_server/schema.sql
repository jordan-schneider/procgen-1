CREATE TABLE IF NOT EXISTS trajectories(
  id INTEGER PRIMARY KEY,
  start_state BLOB NOT NULL,
  actions BLOB NOT NULL,
  length INT NOT NULL,
  env TEXT NOT NULL,
  modality TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS questions(
  id INTEGER PRIMARY KEY,
  first_id INT NOT NULL,
  second_id INT NOT NULL,
  algorithm TEXT NOT NULL,
  env TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS users(
  mturk_id INTEGER PRIMARY KEY NOT NULL,
  site_sequence INT NOT NULL,
  demographics BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS answers(
  id INTEGER PRIMARY KEY,
  user_id INT NOT NULL,
  question_id INT NOT NULL,
  answer INT NOT NULL,
  start_time TEXT NOT NULL,
  end_time TEXT NOT NULL
);