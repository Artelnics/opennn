#pragma once

#include <filesystem>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

namespace tinyxml2 {

enum XMLError {
    XML_SUCCESS = 0,
    XML_NO_ATTRIBUTE,
    XML_WRONG_ATTRIBUTE_TYPE,
    XML_ERROR_FILE_NOT_FOUND,
    XML_ERROR_PARSING
};

class XMLElement;
class XMLDocument;

class XMLAttribute {
public:
    std::string _name;
    std::string _value;
    const char* Name() const { return _name.c_str(); }
    const char* Value() const { return _value.c_str(); }
    const XMLAttribute* Next() const { return _next; }
    XMLAttribute* _next = nullptr;

    int IntValue() const { return std::stoi(_value); }
    float FloatValue() const { return std::stof(_value); }
};

class XMLNode {
public:
    virtual ~XMLNode() = default;
    XMLNode() = default;
    XMLNode(const XMLNode&) = delete;
    XMLNode& operator=(const XMLNode&) = delete;
    XMLNode(XMLNode&&) = default;
    XMLNode& operator=(XMLNode&&) = default;
    virtual XMLElement* ToElement() { return nullptr; }
    const XMLElement* FirstChildElement(const char* name = nullptr) const;
    const XMLElement* NextSiblingElement(const char* name = nullptr) const;

    std::string _value; // Tag name or Text content
    std::vector<std::unique_ptr<XMLElement>> _children;
    XMLNode* _parent = nullptr;
    XMLElement* _next = nullptr;
};

class XMLElement : public XMLNode {
public:
    std::vector<XMLAttribute> _attributes;

    const char* Name() const { return _value.c_str(); }
    XMLElement* ToElement() override { return this; }

    const char* Attribute(const char* name) const {
        for(const auto& a : _attributes) if(a._name == name) return a._value.c_str();
        return nullptr;
    }

    int QueryIntAttribute(const char* name, int* value) const {
        const char* v = Attribute(name);
        if(!v) return 1; // Error
        *value = std::stoi(v);
        return 0; // Success
    }

    int QueryUnsignedAttribute(const char* name, unsigned int* value) const {
        const char* v = Attribute(name);
        if(!v) return 1;
        *value = (unsigned int)std::stoul(v);
        return 0;
    }

    const char* GetText() const;
    XMLElement* DeepClone(XMLDocument* target) const;
};

class XMLDocument : public XMLNode {
public:
    int LoadFile(const char* filename);
    int Parse(const char* xml);
    XMLElement* RootElement() { return _children.empty() ? nullptr : _children[0].get(); }
    XMLElement* NewElement(const char* name);
    void InsertFirstChild(XMLNode* node);
    void InsertEndChild(XMLNode* node);
};

class XMLPrinter {
public:
    std::ostringstream _oss;
    int _indent = 0;

    XMLPrinter(FILE* f = nullptr, bool compact = false, int depth = 0) {}

    void OpenElement(const char* name) {
        _oss << "\n" << std::string(_indent*2, ' ') << "<" << name;
        _stack.push_back(name);
        _indent++;
        _openTag = true;
    }

    void PushAttribute(const char* name, const char* value) {
        _oss << " " << name << "=\"" << value << "\"";
    }

    void PushAttribute(const char* name, int value) { PushAttribute(name, std::to_string(value).c_str()); }

    void PushText(const char* text) {
        if(_openTag) { _oss << ">"; _openTag = false; }
        _oss << text;
    }

    void CloseElement() {
        _indent--;
        if(_openTag) { _oss << "/>"; _openTag = false; }
        else { _oss << "</" << _stack.back() << ">"; }
        _stack.pop_back();
    }

    const char* CStr() const { _buffer = _oss.str(); return _buffer.c_str(); }

private:
    std::vector<std::string> _stack;
    mutable std::string _buffer;
    bool _openTag = false;
};

// OpenNN helper wrappers — writing

void add_xml_element(XMLPrinter& printer, const std::string& name, const std::string& value);
void add_xml_element_attribute(XMLPrinter& printer, const std::string& element_name, const std::string& element_value, const std::string& attribute_name, const std::string& attribute_value);

void write_xml_properties(XMLPrinter& printer, std::initializer_list<std::pair<const char*, std::string>> props);

// OpenNN helper wrappers — reading

float read_xml_type(const XMLElement* root, const std::string& element_name);
long  read_xml_index(const XMLElement* root, const std::string& element_name);
bool  read_xml_bool(const XMLElement* root, const std::string& element_name);
std::string read_xml_string(const XMLElement* root, const std::string& element_name);

std::string read_xml_string_fallback(const XMLElement* root, std::initializer_list<std::string> names);

const XMLElement* require_xml_element(const XMLElement* root, const std::string& element_name);

template<typename Func>
void for_xml_items(const XMLElement* parent, const char* tag, long count, Func func)
{
    const XMLElement* item = parent->FirstChildElement(tag);
    for(long i = 0; i < count; i++)
    {
        if(!item)
            throw std::runtime_error(std::string("Missing XML element: ") + tag + " item " + std::to_string(i + 1));

        func(i, item);
        item = item->NextSiblingElement(tag);
    }
}

XMLDocument load_xml_file(const std::filesystem::path& file_name);
const XMLElement* get_xml_root(const XMLDocument& document, const std::string& tag);

template<typename T>
void print_xml(const T& object)
{
    XMLPrinter printer;
    object.to_XML(printer);
    std::cout << printer.CStr() << std::endl;
}

} // namespace
