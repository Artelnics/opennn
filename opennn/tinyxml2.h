#pragma once

#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

namespace tinyxml2 {

enum XmlError {
    XML_SUCCESS = 0,
    XML_NO_ATTRIBUTE,
    XML_WRONG_ATTRIBUTE_TYPE,
    XML_ERROR_FILE_NOT_FOUND,
    XML_ERROR_PARSING
};

class XmlElement;
class XmlDocument;

class XmlAttribute {
public:
    std::string _name;
    std::string _value;
    const char* name() const { return _name.c_str(); }
    const char* value() const { return _value.c_str(); }
    const XmlAttribute* next() const { return _next; }
    XmlAttribute* _next = nullptr;

    int int_value() const { return std::stoi(_value); }
    float float_value() const { return std::stof(_value); }
};

class XmlNode {
public:
    virtual ~XmlNode() = default;
    XmlNode() = default;
    XmlNode(const XmlNode&) = delete;
    XmlNode& operator=(const XmlNode&) = delete;
    XmlNode(XmlNode&&) = default;
    XmlNode& operator=(XmlNode&&) = default;
    virtual XmlElement* to_element() { return nullptr; }
    const XmlElement* first_child_element(const char* name = nullptr) const;
    const XmlElement* next_sibling_element(const char* name = nullptr) const;

    std::string _value; // Tag name or Text content
    std::vector<std::unique_ptr<XmlElement>> _children;
    XmlNode* _parent = nullptr;
    XmlElement* _next = nullptr;
};

class XmlElement : public XmlNode {
public:
    std::vector<XmlAttribute> _attributes;

    const char* name() const { return _value.c_str(); }
    XmlElement* to_element() override { return this; }

    const char* attribute(const char* name) const {
        for(const auto& a : _attributes) if(a._name == name) return a._value.c_str();
        return nullptr;
    }

    int query_int_attribute(const char* name, int* value) const {
        const char* v = attribute(name);
        if(!v) return 1;
        *value = std::stoi(v);
        return 0;
    }

    int query_unsigned_attribute(const char* name, unsigned int* value) const {
        const char* v = attribute(name);
        if(!v) return 1;
        *value = (unsigned int)std::stoul(v);
        return 0;
    }

    const char* get_text() const;
    XmlElement* deep_clone(XmlDocument* target) const;
};

class XmlDocument : public XmlNode {
public:
    int load_file(const char* filename);
    int parse(const char* xml);
    XmlElement* root_element() { return _children.empty() ? nullptr : _children[0].get(); }
    XmlElement* new_element(const char* name);
    void insert_first_child(XmlNode* node);
    void insert_end_child(XmlNode* node);
};

class XmlPrinter {
public:
    std::ostringstream _oss;
    int _indent = 0;

    XmlPrinter(FILE* f = nullptr, bool compact = false, int depth = 0) {}

    void open_element(const char* name) {
        _oss << "\n" << std::string(_indent*2, ' ') << "<" << name;
        _stack.push_back(name);
        _indent++;
        _openTag = true;
    }

    void push_attribute(const char* name, const char* value) {
        _oss << " " << name << "=\"" << value << "\"";
    }

    void push_attribute(const char* name, int value) { push_attribute(name, std::to_string(value).c_str()); }

    void push_text(const char* text) {
        if(_openTag) { _oss << ">"; _openTag = false; }
        _oss << text;
    }

    void close_element() {
        _indent--;
        if(_openTag) { _oss << "/>"; _openTag = false; }
        else { _oss << "</" << _stack.back() << ">"; }
        _stack.pop_back();
    }

    const char* c_str() const { _buffer = _oss.str(); return _buffer.c_str(); }

private:
    std::vector<std::string> _stack;
    mutable std::string _buffer;
    bool _openTag = false;
};

// OpenNN helper wrappers — writing

void add_xml_element(XmlPrinter& printer, const std::string& name, const std::string& value);
void add_xml_element_attribute(XmlPrinter& printer, const std::string& element_name, const std::string& element_value, const std::string& attribute_name, const std::string& attribute_value);

void write_xml_properties(XmlPrinter& printer, std::initializer_list<std::pair<const char*, std::string>> props);

// OpenNN helper wrappers — reading

float read_xml_type(const XmlElement* root, const std::string& element_name);
long  read_xml_index(const XmlElement* root, const std::string& element_name);
bool  read_xml_bool(const XmlElement* root, const std::string& element_name);
std::string read_xml_string(const XmlElement* root, const std::string& element_name);

std::string read_xml_string_fallback(const XmlElement* root, std::initializer_list<std::string> names);

const XmlElement* require_xml_element(const XmlElement* root, const std::string& element_name);

template<typename Func>
void for_xml_items(const XmlElement* parent, const char* tag, long count, Func func)
{
    const XmlElement* item = parent->first_child_element(tag);
    for(long i = 0; i < count; i++)
    {
        if(!item)
            throw std::runtime_error(std::string("Missing XML element: ") + tag + " item " + std::to_string(i + 1));

        func(i, item);
        item = item->next_sibling_element(tag);
    }
}

XmlDocument load_xml_file(const std::filesystem::path& file_name);
const XmlElement* get_xml_root(const XmlDocument& document, const std::string& tag);

template<typename T>
void print_xml(const T& object)
{
    XmlPrinter printer;
    object.to_XML(printer);
    std::cout << printer.c_str() << std::endl;
}

} // namespace
