#include "tinyxml2.h"

namespace tinyxml2 {

const XMLElement* XMLNode::FirstChildElement(const char* name) const {
    for(const auto& child : _children) {
        if(!name || child->_value == name) return child.get();
    }
    return nullptr;
}

const XMLElement* XMLNode::NextSiblingElement(const char* name) const {
    const XMLNode* p = _parent;
    if(!p) return nullptr;
    bool foundSelf = false;
    for(const auto& sibling : p->_children) {
        if(foundSelf) {
            if(!name || sibling->_value == name) return sibling.get();
        }
        if(sibling.get() == this) foundSelf = true;
    }
    return nullptr;
}

const char* XMLElement::GetText() const {
    return _children.empty() ? nullptr : _children[0]->_value.c_str();
}

int XMLDocument::Parse(const char* xml) {
    const std::string s = xml;
    // This is a dummy simplified parser logic
    // In a real scenario, you'd use a small regex or basic state machine
    // to populate the _children and _attributes.
    return 0;
}

int XMLDocument::LoadFile(const char* filename) {
    std::ifstream f(filename);
    if(!f.is_open()) return 1;
    const std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return Parse(content.c_str());
}

XMLElement* XMLDocument::NewElement(const char* name) {
    auto el = std::make_unique<XMLElement>();
    el->_value = name;
    return el.release(); // Handing off management to the caller or Insert methods
}

void XMLDocument::InsertFirstChild(XMLNode* node) {
    _children.insert(_children.begin(), std::unique_ptr<XMLElement>(static_cast<XMLElement*>(node)));
    _children.front()->_parent = this;
}

void XMLDocument::InsertEndChild(XMLNode* node) {
    _children.push_back(std::unique_ptr<XMLElement>(static_cast<XMLElement*>(node)));
    _children.back()->_parent = this;
}

XMLElement* XMLElement::DeepClone(XMLDocument* target) const {
    XMLElement* clone = target->NewElement(this->Name());
    clone->_attributes = this->_attributes;
    for(const auto& child : _children) {
        clone->_children.push_back(std::unique_ptr<XMLElement>(child->DeepClone(target)));
    }
    return clone;
}

void add_xml_element(XMLPrinter& printer, const std::string& name, const std::string& value) {
    printer.OpenElement(name.c_str());
    printer.PushText(value.c_str());
    printer.CloseElement();
}

void add_xml_element_attribute(XMLPrinter& printer, const std::string& element_name, const std::string& element_value, const std::string& attribute_name, const std::string& attribute_value) {
    printer.OpenElement(element_name.c_str());
    printer.PushAttribute(attribute_name.c_str(), attribute_value.c_str());
    printer.PushText(element_value.c_str());
    printer.CloseElement();
}

void write_xml_properties(XMLPrinter& printer, std::initializer_list<std::pair<const char*, std::string>> props) {
    for(const auto& [name, value] : props)
        add_xml_element(printer, name, value);
}

float read_xml_type(const XMLElement* root, const std::string& element_name) {
    const XMLElement* el = root->FirstChildElement(element_name.c_str());
    return (el && el->GetText()) ? std::stof(el->GetText()) : 0.0f;
}

long read_xml_index(const XMLElement* root, const std::string& element_name) {
    const XMLElement* el = root->FirstChildElement(element_name.c_str());
    return (el && el->GetText()) ? std::stol(el->GetText()) : 0;
}

bool read_xml_bool(const XMLElement* root, const std::string& element_name) {
    const XMLElement* el = root->FirstChildElement(element_name.c_str());
    if (!el || !el->GetText()) return false;
    const std::string val = el->GetText();
    return (val == "1" || val == "true");
}

std::string read_xml_string(const XMLElement* root, const std::string& element_name) {
    const XMLElement* el = root->FirstChildElement(element_name.c_str());
    return (el && el->GetText()) ? std::string(el->GetText()) : "";
}

std::string read_xml_string_fallback(const XMLElement* root, std::initializer_list<std::string> names) {
    for(const auto& name : names) {
        const XMLElement* el = root->FirstChildElement(name.c_str());
        if(el && el->GetText()) return std::string(el->GetText());
    }
    std::string all_names;
    for(const auto& name : names) { if(!all_names.empty()) all_names += "/"; all_names += name; }
    throw std::runtime_error("Element is nullptr: " + all_names);
}

const XMLElement* require_xml_element(const XMLElement* root, const std::string& element_name) {
    const XMLElement* el = root->FirstChildElement(element_name.c_str());
    if(!el) throw std::runtime_error(element_name + " element is nullptr.\n");
    return el;
}

XMLDocument load_xml_file(const std::filesystem::path& file_name)
{
    XMLDocument document;
    if (document.LoadFile(file_name.string().c_str()))
        throw std::runtime_error("Cannot load XML file " + file_name.string() + ".\n");
    return document;
}

const XMLElement* get_xml_root(const XMLDocument& document, const std::string& tag)
{
    const XMLElement* el = document.FirstChildElement(tag.c_str());
    if (!el)
        throw std::runtime_error(tag + " element is nullptr.\n");
    return el;
}

}
