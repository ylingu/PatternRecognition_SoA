export module thread_pool;

import std;

export class ThreadPool {
public:
    explicit ThreadPool(
        size_t thread_count = std::thread::hardware_concurrency());
    ~ThreadPool();

    template <typename F, typename... Args>
        requires std::invocable<F, Args...>
    auto enqueue(F&& func, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>>;

private:
    void worker_loop(std::stop_token stoken);

    std::vector<std::jthread> workers_;
    std::queue<std::move_only_function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable_any cv_;
    std::stop_source stop_source_;
};

template <typename F, typename... Args>
    requires std::invocable<F, Args...>
auto ThreadPool::enqueue(F&& func, Args&&... args)
    -> std::future<std::invoke_result_t<F, Args...>> {
    using ReturnType = std::invoke_result_t<F, Args...>;

    auto task = std::packaged_task<ReturnType()>(
        [func = std::forward<F>(func),
         ... args = std::forward<Args>(args)]() mutable {
            return std::invoke(std::move(func), std::move(args)...);
        });

    auto result = task.get_future();
    {
        std::unique_lock lock{queue_mutex_};
        tasks_.emplace(std::move(task));
    }
    cv_.notify_one();
    return result;
}

ThreadPool::ThreadPool(size_t thread_count) : stop_source_{} {
    workers_.reserve(thread_count);
    for (size_t i = 0; i < thread_count; ++i) {
        workers_.emplace_back(
            [this](std::stop_token stoken) { worker_loop(stoken); },
            stop_source_.get_token());
    }
}

ThreadPool::~ThreadPool() {
    stop_source_.request_stop();
    cv_.notify_all();
}

void ThreadPool::worker_loop(std::stop_token stoken) {
    while (true) {
        std::move_only_function<void()> task;
        {
            std::unique_lock lock{queue_mutex_};

            if (!cv_.wait(lock, stoken, [this] { return !tasks_.empty(); })) {
                break;
            }

            task = std::move(tasks_.front());
            tasks_.pop();
        }

        task();
    }
}