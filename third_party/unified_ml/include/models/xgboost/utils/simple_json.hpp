#pragma once
//              
//  SimpleJson — minimal recursive-descent JSON parser
//
//  No external dependencies. Supports the exact JSON format produced by
//  GradientBooster::dump_model_json() / DecisionTree::dump_json().
//
//  Supported types: object, array, string, number (double), bool, null.
//
//  Usage:
//  auto root = SimpleJson::parse(json_text);
//  double v = root["base_score"].as_double();
//  for (auto& tree : root["trees"].as_array()) { ... }
//
//  NOTE ON DESIGN
//             
//  JsonValue is a self-referential type: it contains itself recursively
//  (arrays of JsonValue, objects mapping string→JsonValue).
//
//  std::unordered_map<K, JsonValue> as a *member* of JsonValue is ill-formed:
//  the map's internal node type (pair<const K, V>) requires V to be complete
//  at the point of template instantiation, which hasn't happened yet when
//  the member is declared.  GCC 11 rejects this (std::pair<…>::second has
//  incomplete type).
//
//  Fix: store object entries as std::vector<std::pair<std::string, JsonValue>>.
//  C++17 explicitly allows std::vector<T> as a member of T when T is still
//  incomplete (the allocator completeness requirements are deferred to the
//  point of actual use).  Linear-scan lookup is fine for JSON objects of
//  the sizes produced by model serialisation.
//              
#include <string>
#include <vector>
#include <utility>  // std::pair
#include <stdexcept>
#include <cctype>
#include <cstdlib>
#include <memory>

namespace xgb {

struct JsonValue {
  enum class Type { Null, Bool, Number, String, Array, Object };

  Type type{Type::Null};
  double  number{0.0};
  bool    boolean{false};
  std::string str;

  // std::vector<JsonValue>  is fine in C++17 — vector allows incomplete value_type
  std::vector<JsonValue> array;

  // Use vector<pair<string,JsonValue>> instead of unordered_map<string,JsonValue>.
  // unordered_map requires JsonValue to be complete at node instantiation time,
  // which is impossible when it's used as a member of JsonValue itself.
  // vector defers completeness requirements to the first actual use (push_back etc.)
  // which happens only after JsonValue is fully defined.
  std::vector<std::pair<std::string, JsonValue>> object_entries;

  // Constructors          
  JsonValue() = default;
  explicit JsonValue(double d)  : type(Type::Number), number(d)    {}
  explicit JsonValue(bool b)  : type(Type::Bool), boolean(b)   {}
  explicit JsonValue(std::string s) : type(Type::String), str(std::move(s))  {}

  // Type tests           
  bool is_null() const { return type == Type::Null; }
  bool is_number() const { return type == Type::Number; }
  bool is_bool() const { return type == Type::Bool; }
  bool is_string() const { return type == Type::String; }
  bool is_array()  const { return type == Type::Array;  }
  bool is_object() const { return type == Type::Object; }

  // Value accessors         
  double     as_double() const { return number; }
  float      as_float()  const { return static_cast<float>(number); }
  int      as_int()  const { return static_cast<int>(number); }
  bool     as_bool() const { return boolean; }
  const std::string& as_string() const { return str; }

  const std::vector<JsonValue>& as_array() const { return array; }
    std::vector<JsonValue>& as_array()   { return array; }

  // Object key existence        
  bool has(const std::string& key) const {
    if (type != Type::Object) return false;
    for (const auto& kv : object_entries)
    if (kv.first == key) return true;
    return false;
  }

  // Object key access (const)        
  const JsonValue& operator[](const std::string& key) const {
    for (const auto& kv : object_entries)
    if (kv.first == key) return kv.second;
    throw std::runtime_error("JSON key not found: " + key);
  }

  // Object key access (mutating — inserts if missing, like unordered_map)  
  JsonValue& operator[](const std::string& key) {
    for (auto& kv : object_entries)
    if (kv.first == key) return kv.second;
    object_entries.emplace_back(key, JsonValue{});
    return object_entries.back().second;
  }
};

//              
//  Parser — recursive descent over a string cursor
//              
class SimpleJson {
public:
  static JsonValue parse(const std::string& text) {
    SimpleJson p(text);
    p.skip_ws();
    JsonValue v = p.parse_value();
    return v;
  }

private:
  const std::string& text_;
  std::size_t pos_{0};

  explicit SimpleJson(const std::string& t) : text_(t) {}

  void skip_ws() {
    while (pos_ < text_.size() && std::isspace((unsigned char)text_[pos_]))
    ++pos_;
  }

  char peek() const {
    return pos_ < text_.size() ? text_[pos_] : '\0';
  }

  char consume() {
    if (pos_ >= text_.size())
    throw std::runtime_error("Unexpected end of JSON");
    return text_[pos_++];
  }

  void expect(char c) {
    skip_ws();
    char got = consume();
    if (got != c)
    throw std::runtime_error(
      std::string("Expected '") + c + "' got '" + got + "'");
  }

  JsonValue parse_value() {
    skip_ws();
    char c = peek();
    if (c == '{') return parse_object();
    if (c == '[') return parse_array();
    if (c == '"') return parse_string_value();
    if (c == 't' || c == 'f') return parse_bool();
    if (c == 'n') return parse_null();
    return parse_number();
  }

  JsonValue parse_object() {
    expect('{');
    JsonValue v;
    v.type = JsonValue::Type::Object;
    skip_ws();
    if (peek() == '}') { consume(); return v; }
    while (true) {
    skip_ws();
    expect('"');
    std::string key = parse_raw_string();
    expect(':');
    skip_ws();
    v[key] = parse_value(); // uses operator[] which appends to object_entries
    skip_ws();
    char sep = peek();
    if (sep == ',') { consume(); continue; }
    if (sep == '}') { consume(); break; }
    throw std::runtime_error("Expected ',' or '}' in object");
    }
    return v;
  }

  JsonValue parse_array() {
    expect('[');
    JsonValue v;
    v.type = JsonValue::Type::Array;
    skip_ws();
    if (peek() == ']') { consume(); return v; }
    while (true) {
    skip_ws();
    v.array.push_back(parse_value());
    skip_ws();
    char sep = peek();
    if (sep == ',') { consume(); continue; }
    if (sep == ']') { consume(); break; }
    throw std::runtime_error("Expected ',' or ']' in array");
    }
    return v;
  }

  JsonValue parse_string_value() {
    expect('"');
    return JsonValue(parse_raw_string());
  }

  std::string parse_raw_string() {
    std::string result;
    while (pos_ < text_.size()) {
    char c = consume();
    if (c == '"') return result;
    if (c == '\\') {
      char esc = consume();
      switch (esc) {
        case '"':  result += '"';  break;
        case '\\': result += '\\'; break;
        case '/':  result += '/';  break;
        case 'n':  result += '\n'; break;
        case 'r':  result += '\r'; break;
        case 't':  result += '\t'; break;
        default: result += esc;  break;
      }
    } else {
      result += c;
    }
    }
    throw std::runtime_error("Unterminated string in JSON");
  }

  JsonValue parse_number() {
    std::size_t start = pos_;
    if (peek() == '-') ++pos_;
    while (pos_ < text_.size() && std::isdigit((unsigned char)text_[pos_])) ++pos_;
    if (pos_ < text_.size() && text_[pos_] == '.') {
    ++pos_;
    while (pos_ < text_.size() && std::isdigit((unsigned char)text_[pos_])) ++pos_;
    }
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
    ++pos_;
    if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
    while (pos_ < text_.size() && std::isdigit((unsigned char)text_[pos_])) ++pos_;
    }
    std::string s = text_.substr(start, pos_ - start);
    if (s.empty())
    throw std::runtime_error("Expected number at pos " + std::to_string(start));
    return JsonValue(std::stod(s));
  }

  JsonValue parse_bool() {
    if (text_.substr(pos_, 4) == "true")  { pos_ += 4; return JsonValue(true);  }
    if (text_.substr(pos_, 5) == "false") { pos_ += 5; return JsonValue(false); }
    throw std::runtime_error("Expected 'true' or 'false'");
  }

  JsonValue parse_null() {
    if (text_.substr(pos_, 4) == "null") {
    pos_ += 4;
    return JsonValue{}; // default type is Null
    }
    throw std::runtime_error("Expected 'null'");
  }
};

} // namespace xgb
