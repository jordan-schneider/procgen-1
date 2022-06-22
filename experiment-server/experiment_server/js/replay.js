import { post } from './utils.js';

const games = [];
let gameStates = null;
let question = null;
const usedQuestions = [];
let startTime = null;
let stopTime = null;
let questionStarted = false;

function parseOpts() {
  try {
    const search = new URLSearchParams(window.location.search);
    const ret = {};
    for (const [k, v] of search.entries()) {
      ret[k] = JSON.parse(v);
    }
    return ret;
  } catch (e) {
    console.error('Query string is invalid');
    return {};
  }
}

async function requestRandomQuestion({
  env = 'miner', lengths = [10, 10], types = ['traj', 'traj'], excludeIds = [],
} = {}) {
  return post('/random_question', JSON.stringify({
    env,
    lengths,
    types,
    exclude_ids: excludeIds,
  })).then((resp) => resp.json()).then((json) => JSON.parse(json));
}

async function requestQuestionByName(name) {
  return post('/named_question', JSON.stringify({
    name
  })).then((resp) => resp.json()).then((json) => JSON.parse(json));
}

function prepareState(state) {
  const newGrid = new Int32Array(state.grid.length);
  for (const [key, value] of Object.entries(state.grid)) {
    newGrid[parseInt(key, 10)] = value;
  }

  const out = {
    grid: newGrid,
    grid_width: state.grid_shape[0],
    grid_height: state.grid_shape[1],
    agent_x: state.agent_pos[0],
    agent_y: state.agent_pos[1],
    exit_x: state.exit_pos[0],
    exit_y: state.exit_pos[1],
  };
  out.grid = newGrid;

  return out;
}

function makeGameState(game, traj) {
  return {
    game,
    traj,
    playState: 'paused',
    time: 0,
  };
}



async function parseQuestion() {
  const leftTraj = question.trajs[0];
  const rightTraj = question.trajs[1];
  leftTraj.start_state = prepareState(leftTraj.start_state);
  rightTraj.start_state = prepareState(rightTraj.start_state);

  games[0].setState(leftTraj.start_state);
  games[0].render();
  games[1].setState(rightTraj.start_state);
  games[1].render();

  gameStates = [makeGameState(games[0], leftTraj), makeGameState(games[1], rightTraj)];
}

function checkStep(state) {
  const { game } = state;
  if (state.playState !== 'paused') {
    const { time } = state;
    const { actions } = state.traj;
    if (time < actions.length) {
      const leftAction = actions[time];
      game.step(leftAction);
      game.render();
      state.time += 1;
    }
  }
}

async function main() {
  const opts = parseOpts();

  const questionName = opts.questionName;
  delete opts.questionName;

  const leftGame = await CheerpGame.init({
    ...CheerpGame.defaultOpts(),
    ...opts,
  });
  const rightGame = await CheerpGame.init({
    ...CheerpGame.defaultOpts(),
    ...opts,
  });
  leftGame.getState();
  games.push(leftGame, rightGame);
  document.getElementById('leftGame').appendChild(leftGame.getCanvas());
  document.getElementById('rightGame').appendChild(rightGame.getCanvas());

  if (questionName !== undefined) {
    question = await requestQuestionByName(questionName);
    await parseQuestion();
  } else {
    question = await requestRandomQuestion({ excludeIds: usedQuestions });
    await parseQuestion();
  }

  setInterval(() => {
    checkStep(gameStates[0]);
    checkStep(gameStates[1]);
  }, 500);
}

function getSideIndex(side) {
  if (side === 'left') {
    return 0;
  }
  return 1;
}

function pause(side) {
  gameStates[getSideIndex(side)].playState = 'paused';
}
async function pauseLeft() {
  pause('left');
}
async function pauseRight() {
  pause('right');
}

function play(side) {
  if (!questionStarted) {
    startTime = Date.now();
    questionStarted = true;
  }
  gameStates[getSideIndex(side)].playState = 'playing';
}
async function playLeft() {
  play('left');
}
async function playRight() {
  play('right');
}

function restart(side) {
  const index = getSideIndex(side);
  gameStates[index].time = 0;
  gameStates[index].playState = 'paused';
  games[index].setState(gameStates[index].traj.start_state);
  games[index].render();
}
async function restartLeft() {
  restart('left');
}
async function restartRight() {
  restart('right');
}

async function select(side) {
  stopTime = Date.now();
  post('/submit_answer', JSON.stringify({
    id: question.id,
    answer: side,
    startTime,
    stopTime,
  }));
  questionStarted = false;
  usedQuestions.push(question.id);
  await parseQuestion();
}
async function selectLeft() {
  select('left');
}
async function selectRight() {
  select('right');
}

window.pauseLeft = pauseLeft;
window.pauseRight = pauseRight;
window.playLeft = playLeft;
window.playRight = playRight;
window.restartLeft = restartLeft;
window.restartRight = restartRight;
window.selectLeft = selectLeft;
window.selectRight = selectRight;

main();
