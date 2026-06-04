#pragma once

/**
 * @file logger.hpp
 * @brief Application-controlled logging hooks for XGBoost-style components.
 */

#include <chrono>
#include <functional>
#include <iomanip>
#include <sstream>
#include <string>

namespace xgb {

/** @brief Callback target used to consume formatted log records. */
using LogSink = std::function<void(const std::string&)>;

/** @brief Severity threshold for SDK log records. */
enum class LogLevel { DEBUG = 0, INFO, WARNING, ERROR };

/**
 * @brief Central logger for XGBoost-style components.
 * @note The SDK never writes directly to stdout. Applications can attach a
 * custom sink or rely on the default diagnostic sink.
 */
class Logger {
public:
  /** @brief Access the process-local logger instance. */
  static Logger& instance();

  /** @brief Set the minimum emitted severity level. */
  void set_level(LogLevel level) noexcept { min_level_ = level; }
  /** @brief Silence all log output without changing configured sinks. */
  void set_silent(bool silent) noexcept { silent_ = silent; }
  /** @brief Install an application-owned sink for formatted log messages. */
  void set_sink(LogSink sink) { sink_ = std::move(sink); }

  /**
   * @brief Emit a log record.
   * @param level Record severity.
   * @param msg Message text.
   * @param file Optional source file metadata.
   * @param line Optional source line metadata.
   */
  void log(LogLevel level, const std::string& msg,
     const char* file = nullptr, int line = 0);

  /** @brief Emit a debug-level record. */
  void debug(const std::string& msg) { log(LogLevel::DEBUG, msg); }
  /** @brief Emit an info-level record. */
  void info(const std::string& msg) { log(LogLevel::INFO,  msg); }
  /** @brief Emit a warning-level record. */
  void warning(const std::string& msg) { log(LogLevel::WARNING, msg); }
  /** @brief Emit an error-level record. */
  void error(const std::string& msg) { log(LogLevel::ERROR, msg); }

private:
  Logger() = default;
  LogLevel min_level_{LogLevel::INFO};
  bool silent_{false};
  LogSink sink_{};

  static const char* level_str(LogLevel l);
  static std::string timestamp();
};

//        
//  Convenience macros
//        
#define XGB_LOG_DEBUG(msg)             \
  ::xgb::Logger::instance().log(         \
    ::xgb::LogLevel::DEBUG, (msg), __FILE__, __LINE__)

#define XGB_LOG_INFO(msg)            \
  ::xgb::Logger::instance().log(         \
    ::xgb::LogLevel::INFO, (msg), __FILE__, __LINE__)

#define XGB_LOG_WARN(msg)            \
  ::xgb::Logger::instance().log(         \
    ::xgb::LogLevel::WARNING, (msg), __FILE__, __LINE__)

#define XGB_LOG_ERROR(msg)             \
  ::xgb::Logger::instance().log(         \
    ::xgb::LogLevel::ERROR, (msg), __FILE__, __LINE__)

// Stream-style helper      
struct LogStream {
  std::ostringstream oss;
  LogLevel level;
  explicit LogStream(LogLevel l) : level(l) {}
  ~LogStream() { Logger::instance().log(level, oss.str()); }
  template<typename T>
  LogStream& operator<<(const T& v) { oss << v; return *this; }
};

#define XGB_INFO  ::xgb::LogStream(::xgb::LogLevel::INFO)
#define XGB_DEBUG ::xgb::LogStream(::xgb::LogLevel::DEBUG)

} // namespace xgb
