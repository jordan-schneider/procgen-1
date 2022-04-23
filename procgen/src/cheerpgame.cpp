#include <cheerp/client.h>
#include "game-registry.h"
#include "game.h"
#include "resources.h"
#include "state.h"
#include "keyboard.h"
#include "vecoptions.h"

std::vector<std::vector<const char*>> combos =
{
	{"ArrowLeft", "ArrowDown"},
	{"ArrowLeft",},
	{"ArrowLeft", "ArrowUp"},
	{"ArrowDown",},
	{},
	{"ArrowUp",},
	{"ArrowRight", "ArrowDown"},
	{"ArrowRight",},
	{"ArrowRight", "ArrowUp"},
	{"KeyD",},
	{"KeyA",},
	{"KeyW",},
	{"KeyS",},
	{"KeyQ",},
	{"KeyE",},
};

// we want system independent hashing. std::hash doesn't give this.
inline uint32_t hash_str_uint32(const std::string &str) {
	uint32_t hash = 0x811c9dc5;
	uint32_t prime = 0x1000193;

	for (size_t i = 0; i < str.size(); i++) {
		uint8_t value = str[i];
		hash = hash ^ value;
		hash *= prime;
	}

	return hash;
}

class [[cheerp::jsexport]] CheerpGame {
public:
	static client::Object* defaultOpts()
	{
		client::Object* ret;
		__asm__("{\
			resource_root:'data/assets/',\
			env_name:'miner',\
			rand_seed:0,\
			use_generated_assets:false,\
			start_level:0,\
			distribution_mode: 1,\
		}" : "=r"(ret));
		return ret;
	}
	void step()
	{
		game->action = getAction();
		game->step();
		game->observe();
		totalReward+= state.reward;
		displayState();
		kb->clear();
	}
	void displayState()
	{
		std::string hud = std::string("Reward: ") + std::to_string(state.reward) + "\n" +
			"Total reward: " + std::to_string(totalReward) + "\n" +
			"Level seed: " + std::to_string(state.level_seed) + "\n" +
			"Previous level seed: " + std::to_string(state.prev_level_seed) + "\n" +
			"Previous level complete: " + std::to_string(state.prev_level_complete) + "\n" +
			"Done: " + std::to_string(state.done) + "\n";
		stateDiv->set_innerText(hud.c_str());
	}
	void destroy()
	{
		delete kb;
		kb = nullptr;
		delete game;
		game = nullptr;
	}
	static client::Promise* init(client::Object* opts)
	{
		if (opts == nullptr)
			opts = defaultOpts();
		VecOptions vopts(opts);
		std::string resource_root;

		vopts.consume_string("resource_root", &resource_root);
		auto* p = images_load(resource_root);
		return p->then(cheerp::Callback([vopts = std::move(vopts)]() mutable {
			return new CheerpGame(vopts);
		}));
	}

private:
	CheerpGame(VecOptions& opts)
	{
		// Setup the dom
		auto* div = client::document.getElementById("app");
		kb = new Keyboard(client::document.get_body());
		canvas = static_cast<client::HTMLCanvasElement*>(client::document.createElement("canvas"));
		canvas->set_width(RENDER_RES);
		canvas->set_height(RENDER_RES);
		div->appendChild(canvas);
		stateDiv = client::document.createElement("div");
		div->appendChild(stateDiv);

		// Setup the game
		initGame(opts);

		// First step
		step();

		// Following steps
		client::setInterval(cheerp::Callback([this]()
		{
			step();
		}), 100);
	}

	void initGame(VecOptions& opts)
	{
		std::string env_name;

		int num_levels = 0;
		int start_level = -1;

		int rand_seed = 0;
		std::string resource_root;

		opts.consume_string("env_name", &env_name);
		opts.consume_int("num_levels", &num_levels);
		opts.consume_int("start_level", &start_level);
		opts.consume_int("rand_seed", &rand_seed);
		opts.consume_string("resource_root", &resource_root);

		fassert(env_name != "");
		fassert(num_levels >= 0);
		fassert(start_level >= 0);

		int level_seed_low = 0;
		int level_seed_high = 0;

		if (num_levels == 0) {
			level_seed_low = 0;
			level_seed_high = INT32_MAX;
		} else if (num_levels > 0) {
			level_seed_low = start_level;
			level_seed_high = start_level + num_levels;
		}

		RandGen game_level_seed_gen;
		game_level_seed_gen.seed(rand_seed);

		game = globalGameRegistry->at(env_name)();
		game->set_canvas(canvas);
		game->state = &state;
		fassert(game->game_name == env_name);
		game->level_seed_rand_gen.seed(game_level_seed_gen.randint());
		game->level_seed_high = level_seed_high;
		game->level_seed_low = level_seed_low;
		game->game_n = 0;
		game->is_waiting_for_step = false;
		game->parse_options(env_name, opts);

		// Auto-selected a fixed_asset_seed if one wasn't specified on
		// construction
		if (game->fixed_asset_seed == 0) {
			auto hashed = hash_str_uint32(env_name);
			game->fixed_asset_seed = int(hashed);
		}

		game->game_init();

		game->reset();
		game->observe();
		game->initial_reset_complete = true;
	}

	int getAction()
	{
		int longest = -1;
		int action = -1;
		int i = 0;
		for (auto& combo: combos)
		{
			bool hit = true;
			for (auto* k: combo)
			{
				if (!kb->isPressed(k))
				{
					hit = false;
					break;
				}
			}
			if (hit && longest < (int)combo.size())
			{
				longest = combo.size();
				action = i;
			}
			i++;
		}
		return action;
	}

	Game* game;
	Keyboard* kb;
	GameState state;
	double totalReward{0};
	client::HTMLElement* stateDiv;
	client::HTMLCanvasElement* canvas;
};

extern "C" void __call_exitprocs()
{
	__asm__("throw new Error('Cheerp exited with an error')");
}
extern "C" void _exit()
{
	__asm__("throw new Error('Cheerp exited with an error')");
}
