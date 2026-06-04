#include "models/xgboost/utils/logger.hpp"
#include <ctime>
#include <iostream>
#include <mutex>

namespace xgb {

static std::mutex g_log_mutex;

Logger& Logger::instance() {
  static Logger inst;
  return inst;
}

void Logger::log(LogLevel level, const std::string& msg,
       const char* /*file*/, int /*line*/)
{
  if (silent_ || level < min_level_) return;
  std::lock_guard<std::mutex> lock(g_log_mutex);
  std::ostringstream stream;
  stream << "[" << timestamp() << "] " << level_str(level) << " " << msg;
  if (sink_) {
    sink_(stream.str());
    return;
  }
  std::clog << stream.str() << '\n';
}

const char* Logger::level_str(LogLevel l) {
  switch (l) {
    case LogLevel::DEBUG: return "DEBUG  ";
    case LogLevel::INFO:  return "INFO ";
    case LogLevel::WARNING: return "WARNING";
    case LogLevel::ERROR: return "ERROR  ";
  }
  return "???";
}

std::string Logger::timestamp() {
  auto now = std::chrono::system_clock::now();
  auto t   = std::chrono::system_clock::to_time_t(now);
  std::tm tm_buf{};
#ifdef _WIN32
  localtime_s(&tm_buf, &t);
#else
  localtime_r(&t, &tm_buf);
#endif
  char buf[20];
  std::strftime(buf, sizeof(buf), "%H:%M:%S", &tm_buf);
  return buf;
}

} // namespace xgb
