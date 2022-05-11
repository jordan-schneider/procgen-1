#pragma once

#include <cheerp/client.h>

namespace client {

class GameState : public Object {
  public:
    double get_reward();
    void set_reward(double);
    int get_prev_level_seed();
    void set_prev_level_seed(int);
    int get_level_seed();
    void set_level_seed(int);
    bool get_prev_level_complete();
    void set_prev_level_complete(bool);
    bool get_done();
    void set_done(bool);
    client::HTMLCanvasElement *get_rgb();
    void set_rgb(client::HTMLCanvasElement *);
};

class MinerState : public GameState {
  public:
    void set_grid(client::Int32Array *);
    client::Int32Array *get_grid() const;
    void set_grid_width(int);
    int get_grid_width() const;
    void set_grid_height(int);
    int get_grid_height() const;
    void set_agent_x(int);
    int get_agent_x() const;
    void set_agent_y(int);
    int get_agent_y() const;
    void set_exit_x(int);
    int get_exit_x() const;
    void set_exit_y(int);
    int get_exit_y() const;
};

class MazeState : public GameState {
  public:
    void set_grid(client::Int32Array *);
    void set_grid_width(int);
    void set_grid_height(int);
    void set_agent_x(int);
    void set_agent_y(int);
};
} // namespace client
