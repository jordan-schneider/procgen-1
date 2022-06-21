#include "../basic-abstract-game.h"
#include "../assetgen.h"
#include "../cheerputils.cpp"
#include <set>
#include <queue>
#include <iterator>
const std::string NAME = "miner";

const float COMPLETION_BONUS = 10.0;
const float DIAMOND_REWARD = 1.0;

const int BOULDER = 1;
const int DIAMOND = 2;
const int MOVING_BOULDER = 3;
const int MOVING_DIAMOND = 4;
const int ENEMY = 5;
const int EXIT = 6;
const int DIRT = 9;
const int MUD = 11;
const int DEAD_PLAYER = 12;

const int OOB_WALL = 10;

class MinerGame : public BasicAbstractGame {
  public:
    int diamonds_remaining = -1;
    bool died = false;

    MinerGame()
        : BasicAbstractGame(NAME) {
        main_width = 20;
        main_height = 20;

        mixrate = .5;
        maxspeed = .5;
        has_useful_vel_info = false;

        out_of_bounds_object = OOB_WALL;
        visibility = 8.0;
    }

    void load_background_images() override {
        main_bg_images_ptr = &caves;
    }

    void asset_for_type(int type, std::vector<std::string> &names) override {
        if (type == PLAYER) {
            names.push_back("misc_assets/robot_greenDrive1.png");
        } else if (type == DEAD_PLAYER) {
            names.push_back("misc_assets/fire_1.png");
        } else if (type == BOULDER) {
            names.push_back("misc_assets/elementStone007.png");
        } else if (type == DIAMOND) {
            names.push_back("misc_assets/gemBlue.png");
        } else if (type == EXIT) {
            names.push_back("misc_assets/window.png");
        } else if (type == DIRT) {
            names.push_back("misc_assets/dirt.png");
        } else if (type == MUD) {
            names.push_back("misc_assets/groundB.png");
        } else if (type == OOB_WALL) {
            names.push_back("misc_assets/tile_bricksGrey.png");
        }
    }

    bool is_blocked(const std::shared_ptr<Entity> &src, int target, bool is_horizontal) override {
        if (BasicAbstractGame::is_blocked(src, target, is_horizontal))
            return true;
        if (src->type == PLAYER && (target == BOULDER || target == MOVING_BOULDER || target == OOB_WALL))
            return true;

        return false;
    }

    bool will_reflect(int src, int target) override {
        return BasicAbstractGame::will_reflect(src, target) || (src == ENEMY && (target == BOULDER || target == DIAMOND || target == MOVING_BOULDER || target == MOVING_DIAMOND || target == out_of_bounds_object));
    }

    void handle_agent_collision(const std::shared_ptr<Entity> &obj) override {
        BasicAbstractGame::handle_agent_collision(obj);

        if (obj->type == ENEMY) {
            step_data.done = true;
        } else if (obj->type == EXIT) {
            if (diamonds_remaining == 0) {
                step_data.reward += COMPLETION_BONUS;
                step_data.level_complete = true;
                step_data.done = true;
            }
        }
    }

    int image_for_type(int type) override {
        if (type == MOVING_BOULDER) {
            return BOULDER;
        } else if (type == MOVING_DIAMOND) {
            return DIAMOND;
        }

        return BasicAbstractGame::image_for_type(type);
    }

    int get_agent_index() {
        return int(agent->y) * main_width + int(agent->x);
    }

    void set_action_xy(int move_action) override {
        BasicAbstractGame::set_action_xy(move_action);
        if (action_vx != 0)
            action_vy = 0;
    }

    void choose_new_vel(const std::shared_ptr<Entity> &ent) {
        int is_horizontal = rand_gen.randbool();
        int vel = rand_gen.randn(2) * 2 - 1;
        if (is_horizontal) {
            ent->vx = vel;
            ent->vy = 0;
        } else {
            ent->vx = 0;
            ent->vy = vel;
        }
    }

    void choose_world_dim() override {
        int dist_diff = options.distribution_mode;

        if (dist_diff == EasyMode) {
            main_width = 10;
            main_height = 10;
        } else if (dist_diff == HardMode) {
            main_width = 20;
            main_height = 20;
        } else if (dist_diff == MemoryMode) {
            main_width = 35;
            main_height = 35;
        }
    }

    void game_reset() override {
        BasicAbstractGame::game_reset();

        died = false;

        agent->rx = .5;
        agent->ry = .5;

        int main_area = main_height * main_width;

        options.center_agent = options.distribution_mode == MemoryMode;
        grid_step = true;

        float diamond_pct = 12 / 400.0f;
        float boulder_pct = 80 / 400.0f;
        float mud_pct = 12 / 400.0f;

        int num_diamonds = (int)(diamond_pct * grid_size);
        int num_boulders = (int)(boulder_pct * grid_size);
        int num_mud = (int)(mud_pct * grid_size);

        std::vector<int> obj_idxs = rand_gen.simple_choose(main_area, num_diamonds + num_boulders + num_mud + 1);

        int agent_x = obj_idxs[0] % main_width;
        int agent_y = obj_idxs[0] / main_width;

        agent->x = agent_x + .5;
        agent->y = agent_y + .5;

        for (int i = 0; i < main_area; i++) {
            set_obj(i, DIRT);
        }

        for (int i = 0; i < num_diamonds; i++) {
            int cell = obj_idxs[i + 1];
            set_obj(cell, DIAMOND);
        }

        for (int i = 0; i < num_boulders; i++) {
            int cell = obj_idxs[i + 1 + num_diamonds];
            set_obj(cell, BOULDER);
        }

        for (int i = 0; i < num_mud; i++) {
            int cell = obj_idxs[i + 1 + num_diamonds + num_boulders];
            set_obj(cell, MUD);
        }

        std::vector<int> dirt_cells = get_cells_with_type(DIRT);

        set_obj(int(agent->x), int(agent->y), SPACE);

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int ox = agent_x + i;
                int oy = agent_y + j;
                if (get_obj(ox, oy) == BOULDER) {
                    set_obj(ox, oy, DIRT);
                }
            }
        }

        std::vector<int> exit_candidates;

        for (int cell : dirt_cells) {
            int above_obj = get_obj(cell + main_width);
            if (above_obj == DIRT || above_obj == out_of_bounds_object) {
                exit_candidates.push_back(cell);
            }
        }

        fassert(exit_candidates.size() > 0);

        int exit_cell = exit_candidates[rand_gen.randn((int)(exit_candidates.size()))];
        set_obj(exit_cell, SPACE);
        auto exit = add_entity((exit_cell % main_width) + .5, (exit_cell / main_width) + .5, 0, 0, .5, EXIT);
        exit->render_z = -1;
    }

    int get_moving_type(int type) {
        if (type == DIAMOND)
            return MOVING_DIAMOND;
        if (type == BOULDER)
            return MOVING_BOULDER;

        return type;
    }

    bool is_moving(int type) {
        return type == MOVING_BOULDER || type == MOVING_DIAMOND;
    }

    int get_stationary_type(int type) {
        if (type == MOVING_DIAMOND)
            return DIAMOND;
        if (type == MOVING_BOULDER)
            return BOULDER;

        return type;
    }

    bool is_free(int idx) {
        return get_obj(idx) == SPACE && (get_agent_index() != idx);
    }

    bool is_round(int type) {
        return type == BOULDER || type == MOVING_BOULDER || type == DIAMOND || type == MOVING_DIAMOND;
    }

    void handle_push(Grid<int> &next_grid) {
        int agent_idx = get_agent_index();
        int agentx = agent_idx % main_width;

        if (action_vx == 1 && (agent->vx == 0) && (agentx < main_width - 2) && get_obj(agent_idx + 1) == BOULDER && get_obj(agent_idx + 2) == SPACE) {
            next_grid.set_index(agent_idx + 1, SPACE);
            set_obj(agent_idx + 1, SPACE);
            next_grid.set_index(agent_idx + 2, BOULDER);
            agent->x += 1;
        } else if (action_vx == -1 && (agent->vx == 0) && (agentx > 1) && get_obj(agent_idx - 1) == BOULDER && get_obj(agent_idx - 2) == SPACE) {
            next_grid.set_index(agent_idx - 1, SPACE);
            set_obj(agent_idx - 1, SPACE);
            next_grid.set_index(agent_idx - 2, BOULDER);
            agent->x -= 1;
        }
    }

    void game_step() override {
        BasicAbstractGame::game_step();

        Grid<int> next_grid = get_grid();

        if (died) {
            step_data.done = true;
            return;
        }

        if (action_vx > 0)
            agent->is_reflected = false;
        if (action_vx < 0)
            agent->is_reflected = true;

        handle_push(next_grid);

        int agent_obj = get_obj(int(agent->x), int(agent->y));

        if (agent_obj == DIAMOND) {
            step_data.reward += DIAMOND_REWARD;
        }

        if (agent_obj == DIRT || agent_obj == MUD || agent_obj == DIAMOND) {
            set_obj(int(agent->x), int(agent->y), SPACE);
            next_grid.set(int(agent->x), int(agent->y), SPACE);
        }

        int main_area = main_width * main_height;

        int diamonds_count = 0;
        for (int idx = 0; idx < main_area; idx++) {
            int obj = get_obj(idx);
            int obj_x = idx % main_width;
            int stat_type = get_stationary_type(obj);

            int agent_idx = get_agent_index();

            if (stat_type == DIAMOND) {
                diamonds_count++;
            }

            if (obj == BOULDER || obj == MOVING_BOULDER || obj == DIAMOND || obj == MOVING_DIAMOND) {
                int below_idx = idx - main_width;
                int below_object = get_obj(below_idx);
                bool agent_is_below = agent_idx == below_idx;

                if (below_object == SPACE && !agent_is_below) {
                    next_grid.set_index(idx, SPACE);
                    int two_below_idx = below_idx - main_width;
                    int two_below_obj = get_obj(two_below_idx);
                    int obj_type = two_below_obj == SPACE ? get_moving_type(obj) : stat_type;
                    next_grid.set_index(below_idx, obj_type);
                } else if (agent_is_below && is_moving(obj)) {
                    died = true;
                    // remove(entities.begin(), entities.end(), agent);
                    entities.erase(entities.begin());
                    next_grid.set_index(below_idx, DEAD_PLAYER);
                } else if (is_round(below_object) && obj_x > 0 && is_free(idx - 1) && is_free(idx - main_width - 1)) {
                    next_grid.set_index(idx, SPACE);
                    next_grid.set_index(idx - 1, stat_type);
                } else if (is_round(below_object) && obj_x < main_width - 1 && is_free(idx + 1) && is_free(idx - main_width + 1)) {
                    next_grid.set_index(idx, SPACE);
                    next_grid.set_index(idx + 1, stat_type);
                } else {
                    next_grid.set_index(idx, stat_type);
                }
            }
        }
        for (int idx = 0; idx < main_area; idx++) {
            set_obj(idx, next_grid.get_index(idx));
        }

        diamonds_remaining = diamonds_count;

        for (auto ent : entities) {
            if (ent->type == ENEMY) {
                if (rand_gen.randn(6) == 0) {
                    choose_new_vel(ent);
                }
            }
        }
    }

    void serialize(WriteBuffer *b) override {
        BasicAbstractGame::serialize(b);
        b->write_int(diamonds_remaining);
    }

    void deserialize(ReadBuffer *b) override {
        BasicAbstractGame::deserialize(b);
        diamonds_remaining = b->read_int();
    }

    struct MinerState {
        int grid_width;
        int grid_height;
        std::vector<int> grid;
        int agent_x;
        int agent_y;
        int exit_x;
        int exit_y;
    };

    MinerState
    get_latent_state() {
        MinerState state;

        Grid<int> grid = get_grid();

        state.grid_width = grid.w;
        state.grid_height = grid.h;
        state.grid = grid.data;
        state.agent_x = int(agent->x);
        state.agent_y = int(agent->y);

        std::shared_ptr<Entity> exit_entity = *std::find_if(entities.begin(), entities.end(), [](std::shared_ptr<Entity> e) { return e->type == EXIT; });

        state.exit_x = int(exit_entity->x);
        state.exit_y = int(exit_entity->y);

        return state;
    }

    void observe() override {
        Game::observe();

        auto latent_state = get_latent_state();

        auto *js_state = static_cast<client::MinerState *>(this->state);

        js_state->set_grid_width(latent_state.grid_width);
        js_state->set_grid_height(latent_state.grid_height);

        int32_t grid_size = latent_state.grid_width * latent_state.grid_height;
        int32_t *grid = new int32_t[grid_size];
        auto grid_start = latent_state.grid.begin();
        auto grid_stop = latent_state.grid.begin();
        std::advance(grid_stop, grid_size);
        std::copy(grid_start, grid_stop, grid);
        js_state->set_grid(cheerp::MakeTypedArray(grid, grid_size));

        js_state->set_agent_x(latent_state.agent_x);
        js_state->set_agent_y(latent_state.agent_y);

        js_state->set_exit_x(latent_state.exit_x);
        js_state->set_exit_y(latent_state.exit_y);
    }

    void game_set_state(client::GameState *state) override {
        auto miner_state = static_cast<client::MinerState *>(state);
        auto grid_vals = miner_state->get_grid();

        for (int idx = 0; idx < miner_state->get_grid_width() * miner_state->get_grid_height(); idx++) {
            set_obj(idx, (*grid_vals)[idx]);
        }

        agent->x = miner_state->get_agent_x() + 0.5f;
        agent->y = miner_state->get_agent_y() + 0.5f;

        auto exit = *std::find_if(entities.begin(), entities.end(), [](std::shared_ptr<Entity> e) { return e->type == EXIT; });

        exit->x = miner_state->get_exit_x() + 0.5f;
        exit->y = miner_state->get_exit_y() + 0.5f;
    }
};

REGISTER_GAME(NAME, MinerGame);
