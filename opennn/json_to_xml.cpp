//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   J S O N   T O   X M L   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "json_to_xml.h"

using namespace  opennn;

namespace opennn
{
/// @todo
/*
int JsonToXml::guessPrecision(double& val)
{
    double junk;
    int precision = 0;
    for(;!qFuzzyIsNull(modf(val,&junk));++precision)
        val*=10.0;
    return precision;
}

void JsonToXml::jsonValueToXml(const QJsonValue& val, const QString& name, QXmlStreamWriter& writer)
{
    if(val.isNull() || val.isUndefined())
        return;
    if(val.isObject())
        return jsonObjectToXml(val.toObject(),name,writer);
    if(val.isArray())
        return jsonArrayToXml(val.toArray(),name,writer);
     writer.writeStartElement(name);
     writer.writeAttribute(QStringLiteral("type"),QStringLiteral("value"));
     if(val.isBool()){
         writer.writeAttribute(QStringLiteral("valueType"),QStringLiteral("boolean"));
         writer.writeCharacters(val.toBool() ? QStringLiteral("true") : QStringLiteral("false"));
     }
     else if(val.isDouble()){
         writer.writeAttribute(QStringLiteral("valueType"),QStringLiteral("double"));
         double doubleVal = val.toDouble();
         writer.writeCharacters(QString::number(doubleVal,'f',guessPrecision(doubleVal)));
     }
     else{
         writer.writeAttribute(QStringLiteral("valueType"),QStringLiteral("string"));
         writer.writeCharacters(val.toString());
     }
     writer.writeEndElement();
}

void JsonToXml::jsonObjectToXml(const QJsonObject& obj, const QString& name, QXmlStreamWriter& writer )
{
    if(obj.isEmpty()) return;

    writer.writeStartElement(name);
    writer.writeAttribute(QStringLiteral("type"),QStringLiteral("object"));

    for(auto i = obj.constBegin(), objEnd = obj.constEnd(); i != objEnd; ++i)
    {
        jsonValueToXml(*i,i.key(),writer);
    }
    writer.writeEndElement();
}

void JsonToXml::jsonArrayToXml(const QJsonArray& arr, const QString& name, QXmlStreamWriter& writer )
{
    writer.writeStartElement(name);
    writer.writeAttribute(QStringLiteral("type"),QStringLiteral("array"));

    for(int i = 0, arrSize = arr.size(); i < arrSize;++i)
    {
        jsonValueToXml(arr.at(i),name + QString::number(i),writer);
    }

    writer.writeEndElement();
}

QString JsonToXml::jsonToXml(const QJsonDocument& doc, const QString& rootElementName)
{
    QString result;

    if(doc.isEmpty() || doc.isNull()) return result;

    QXmlStreamWriter writer(&result);
    writer.writeStartDocument();

    if(doc.isArray())
    {
        jsonArrayToXml(doc.array(),rootElementName,writer);
    }
    else if(doc.isObject())
    {
        jsonObjectToXml(doc.object(),rootElementName,writer);
    }

    return result;
}

static QJsonDocument qstringToJson(const QString& path)
{
    QJsonDocument doc = QJsonDocument::fromJson(path.toUtf8());

    return doc;
};
*/
}
