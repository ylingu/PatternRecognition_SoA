#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

/**
 * @class ThreadPool
 * @brief A thread pool for executing tasks in parallel.
 */
class ThreadPool {
public:
    /**
     * @brief Construct a new ThreadPool object
     * @param threads The number of threads in the pool.
     */
    ThreadPool(size_t threads);

    /**
     * @brief Destroy the ThreadPool object and stop all threads.
     */
    ~ThreadPool();

    /**
     * @brief Enqueue a task to be executed by the thread pool.
     * @tparam F The type of the task function.
     * @tparam Args The types of the task function arguments.
     * @param f The task function.
     * @param args The task function arguments.
     * @return A future representing the result of the task.
     */
    template <class F, class... Args>
    auto Enqueue(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result_t<F, Args...>>;

private:
    std::vector<std::thread> workers_;         ///< The worker threads.
    std::queue<std::function<void()>> tasks_;  ///< The tasks queue.
    std::mutex queue_mutex_;  ///< The mutex for synchronizing access to the
                              ///< tasks queue.
    std::condition_variable
        condition_;  ///< The condition variable for notifying worker threads of
                     ///< available tasks.
    bool stop_;      ///< Whether the thread pool is stopping.
};

template <class F, class... Args>
auto ThreadPool::Enqueue(F&& f, Args&&... args)
    -> std::future<typename std::invoke_result_t<F, Args...>> {
    using return_type = typename std::invoke_result_t<F, Args...>;

    // Create a task
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);

        // don't allow enqueueing after stopping the pool
        if (stop_) throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks_.emplace([task]() { (*task)(); });
    }
    condition_.notify_one();
    return res;
}

#endif  // THREADPOOL_H