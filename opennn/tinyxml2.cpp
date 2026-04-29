#include "tinyxml2.h"

namespace tinyxml2 {

const XmlElement* XmlNode::first_child_element(const char* name) const {
    for(const auto& child : _children) {
        if(!name || child->_value == name) return child.get();
    }
    return nullptr;
}

const XmlElement* XmlNode::next_sibling_element(const char* name) const {
    const XmlNode* p = _parent;
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

const char* XmlElement::get_text() const {
    return _children.empty() ? nullptr : _children[0]->_value.c_str();
}

int XmlDocument::parse(const char* xml) {
    if(!xml) return 1;

    _children.clear();

    const char* p = xml;
    std::vector<XmlElement*> stack; // parent stack

    auto skip_ws = [&]() { while(*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p; };

    while(*p) {
        skip_ws();
        if(!*p) break;

        if(*p != '<') {
            // Text content — find the next '<'
            const char* start = p;
            while(*p && *p != '<') ++p;
            if(start != p && !stack.empty()) {
                std::string text(start, p);
                // Trim whitespace
                size_t b = text.find_first_not_of(" \t\n\r");
                size_t e = text.find_last_not_of(" \t\n\r");
                if(b != std::string::npos) {
                    auto text_node = std::make_unique<XmlElement>();
                    text_node->_value = text.substr(b, e - b + 1);
                    text_node->_parent = stack.back();
                    stack.back()->_children.push_back(std::move(text_node));
                }
            }
            continue;
        }

        ++p; // skip '<'

        if(*p == '/') {
            // Closing tag </name>
            ++p;
            while(*p && *p != '>') ++p;
            if(*p == '>') ++p;
            if(!stack.empty()) stack.pop_back();
            continue;
        }

        // Opening tag <name ...>
        skip_ws();
        const char* name_start = p;
        while(*p && *p != ' ' && *p != '>' && *p != '/' && *p != '\n' && *p != '\r' && *p != '\t') ++p;
        std::string tag_name(name_start, p);

        auto element = std::make_unique<XmlElement>();
        element->_value = tag_name;

        // Parse attributes: name="value" pairs
        // The printer format may omit '>' before child elements: <Parent\n  <Child>
        // So we also break on '<' (start of child tag).
        while(*p) {
            skip_ws();
            if(*p == '>' || *p == '/' || *p == '<' || !*p) break;

            // Attribute name
            const char* attr_start = p;
            while(*p && *p != '=' && *p != '>' && *p != '/' && *p != ' ' && *p != '<' && *p != '\n') ++p;
            std::string attr_name(attr_start, p);
            if(attr_name.empty()) break;

            skip_ws();
            if(*p != '=') break; // not an attribute
            ++p; // skip '='
            skip_ws();
            if(*p != '"') break;
            ++p; // skip opening quote
            const char* val_start = p;
            while(*p && *p != '"') ++p;
            std::string attr_value(val_start, p);
            if(*p == '"') ++p;

            XmlAttribute attr;
            attr._name = attr_name;
            attr._value = attr_value;
            element->_attributes.push_back(attr);
        }

        bool self_closing = false;
        if(*p == '/') {
            self_closing = true;
            ++p;
            if(*p == '>') ++p;
        }
        else if(*p == '>') {
            ++p;
        }
        // else: *p == '<' or whitespace — the '>' was omitted (printer format); treat as open tag

        XmlElement* raw = element.get();

        if(stack.empty()) {
            element->_parent = this;
            _children.push_back(std::move(element));
        } else {
            element->_parent = stack.back();
            stack.back()->_children.push_back(std::move(element));
        }

        if(!self_closing) {
            stack.push_back(raw);
        }
    }

    return 0;
}

int XmlDocument::load_file(const char* filename) {
    std::ifstream f(filename);
    if(!f.is_open()) return 1;
    const std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return parse(content.c_str());
}

XmlElement* XmlDocument::new_element(const char* name) {
    auto el = std::make_unique<XmlElement>();
    el->_value = name;
    return el.release();
}

void XmlDocument::insert_first_child(XmlNode* node) {
    _children.insert(_children.begin(), std::unique_ptr<XmlElement>(static_cast<XmlElement*>(node)));
    _children.front()->_parent = this;
}

void XmlDocument::insert_end_child(XmlNode* node) {
    _children.push_back(std::unique_ptr<XmlElement>(static_cast<XmlElement*>(node)));
    _children.back()->_parent = this;
}

XmlElement* XmlElement::deep_clone(XmlDocument* target) const {
    XmlElement* clone = target->new_element(this->name());
    clone->_attributes = this->_attributes;
    for(const auto& child : _children) {
        clone->_children.push_back(std::unique_ptr<XmlElement>(child->deep_clone(target)));
    }
    return clone;
}

void add_xml_element(XmlPrinter& printer, const std::string& name, const std::string& value) {
    printer.open_element(name.c_str());
    printer.push_text(value.c_str());
    printer.close_element();
}

void add_xml_element_attribute(XmlPrinter& printer, const std::string& element_name, const std::string& element_value, const std::string& attribute_name, const std::string& attribute_value) {
    printer.open_element(element_name.c_str());
    printer.push_attribute(attribute_name.c_str(), attribute_value.c_str());
    printer.push_text(element_value.c_str());
    printer.close_element();
}

void write_xml(XmlPrinter& printer, std::initializer_list<std::pair<const char*, std::string>> props) {
    for(const auto& [name, value] : props)
        add_xml_element(printer, name, value);
}

float read_xml_type(const XmlElement* root, const std::string& element_name) {
    const XmlElement* el = root->first_child_element(element_name.c_str());
    return (el && el->get_text()) ? std::stof(el->get_text()) : 0.0f;
}

long read_xml_index(const XmlElement* root, const std::string& element_name) {
    const XmlElement* el = root->first_child_element(element_name.c_str());
    return (el && el->get_text()) ? std::stol(el->get_text()) : 0;
}

bool read_xml_bool(const XmlElement* root, const std::string& element_name) {
    const XmlElement* el = root->first_child_element(element_name.c_str());
    if (!el || !el->get_text()) return false;
    const std::string val = el->get_text();
    return (val == "1" || val == "true");
}

std::string read_xml_string(const XmlElement* root, const std::string& element_name) {
    const XmlElement* el = root->first_child_element(element_name.c_str());
    return (el && el->get_text()) ? std::string(el->get_text()) : "";
}

std::string read_xml_string_fallback(const XmlElement* root, std::initializer_list<std::string> names) {
    for(const auto& name : names) {
        const XmlElement* el = root->first_child_element(name.c_str());
        if(el && el->get_text()) return std::string(el->get_text());
    }
    std::string all_names;
    for(const auto& name : names) { if(!all_names.empty()) all_names += "/"; all_names += name; }
    throw std::runtime_error("Element is nullptr: " + all_names);
}

const XmlElement* require_xml_element(const XmlElement* root, const std::string& element_name) {
    const XmlElement* el = root->first_child_element(element_name.c_str());
    if(!el) throw std::runtime_error(element_name + " element is nullptr.\n");
    return el;
}

XmlDocument load_xml_file(const std::filesystem::path& file_name)
{
    XmlDocument document;
    if (document.load_file(file_name.string().c_str()))
        throw std::runtime_error("Cannot load XML file " + file_name.string() + ".\n");
    return document;
}

const XmlElement* get_xml_root(const XmlDocument& document, const std::string& tag)
{
    const XmlElement* el = document.first_child_element(tag.c_str());
    if (!el)
        throw std::runtime_error(tag + " element is nullptr.\n");
    return el;
}

}
