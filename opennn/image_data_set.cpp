//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I M A G E   D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "image_data_set.h"
#include "images.h"
#include "strings.h"
#include "filesystem"

namespace fs = std::filesystem;
namespace opennn
{


ImageDataSet::ImageDataSet() : DataSet()
{

}


ImageDataSet::ImageDataSet(const Index& new_images_number,
                 const Index& new_height,
                 const Index& new_width,
                 const Index& new_channels_number,
                 const Index& new_targets_number)
{
    set(new_images_number, new_height, new_width, new_channels_number, new_targets_number);
}


/// Returns the number of channels of the images in the data set.

Index ImageDataSet::get_channels_number() const
{
    return channels_number;
}


/// Returns the width of the images in the data set.

Index ImageDataSet::get_image_width() const
{
    return image_width;
}


/// Returns the height of the images in the data set.

Index ImageDataSet::get_image_height() const
{
    return image_height;
}

/// Returns the height of the images in the data set.

Index ImageDataSet::get_image_size() const
{
    return image_height*image_width*channels_number;
}


/// Returns the padding set in the BMP file.

Index ImageDataSet::get_image_padding() const
{
    return padding;
}


bool ImageDataSet::get_augmentation() const
{
    return augmentation;
}


bool ImageDataSet::get_random_reflection_axis_x() const
{
    return random_reflection_axis_x;
}

bool ImageDataSet::get_random_reflection_axis_y() const
{
    return random_reflection_axis_y;
}

type ImageDataSet::get_random_rotation_minimum() const
{
    return random_rotation_minimum;
}

type ImageDataSet::get_random_rotation_maximum() const
{
    return random_rotation_maximum;
}

type ImageDataSet::get_random_horizontal_translation_minimum() const
{
    return random_horizontal_translation_minimum;
}

type ImageDataSet::get_random_horizontal_translation_maximum() const
{
    return random_horizontal_translation_maximum;
}

type ImageDataSet::get_random_vertical_translation_maximum() const
{
    return random_vertical_translation_maximum;
}

type ImageDataSet::get_random_vertical_translation_minimum() const
{
    return random_vertical_translation_minimum;
}



void ImageDataSet::set(const Index& new_images_number,
                  const Index& new_height,
                  const Index& new_width,
                  const Index& new_channels_number,
                  const Index& new_targets_number)
{

    set_image_height(new_height);
    set_image_width(new_width);
    set_channels_number(new_channels_number);

    data_source_path = "";

    const Index new_inputs_number = new_channels_number * new_width * new_height;
    const Index new_variables_number = new_channels_number * new_width * new_height + new_targets_number;

    data.resize(new_images_number, new_variables_number);

    raw_variables.resize(new_variables_number);

    for(Index i = 0; i < new_variables_number; i++)
    {
        if(i < new_inputs_number)
        {
            raw_variables(i).name = "column_" + to_string(i+1);
            raw_variables(i).raw_variable_use = VariableUse::Input;
            raw_variables(i).type = RawVariableType::Numeric;
        }
        else
        {
            raw_variables(i).name = "column_" + to_string(i+1);
            raw_variables(i).raw_variable_use = VariableUse::Target;
            raw_variables(i).type = RawVariableType::Numeric;
        }
    }

    input_variables_dimensions.resize(3);
    input_variables_dimensions.setValues({new_height, new_width, new_channels_number});

    samples_uses.resize(new_images_number);
    split_samples_random();
}


void ImageDataSet::set_channels_number(const int& new_channels_number)
{
    channels_number = new_channels_number;
}


void ImageDataSet::set_image_width(const int& new_width)
{
    image_width = new_width;
}


void ImageDataSet::set_image_height(const int& new_height)
{
    image_height = new_height;
}


void ImageDataSet::set_image_padding(const int& new_padding)
{
    padding = new_padding;
}


void ImageDataSet::set_images_number(const Index & new_images_number)
{
    images_number = new_images_number;
}


void ImageDataSet::set_augmentation(const bool& new_augmentation)
{
    augmentation = new_augmentation;
}


void ImageDataSet::set_random_reflection_axis_x(const bool& new_random_reflection_axis_x)
{
    random_reflection_axis_x = new_random_reflection_axis_x;
}


void ImageDataSet::set_random_reflection_axis_y(const bool& new_random_reflection_axis_y)
{
    random_reflection_axis_y = new_random_reflection_axis_y;
}


void ImageDataSet::set_random_rotation_minimum(const type& new_random_rotation_minimum)
{
    random_rotation_minimum = new_random_rotation_minimum;
}


void ImageDataSet::set_random_rotation_maximum(const type& new_random_rotation_maximum)
{
    random_rotation_maximum = new_random_rotation_maximum;
}


void ImageDataSet::set_random_horizontal_translation_maximum(const type& new_random_horizontal_translation_maximum)
{
    random_horizontal_translation_maximum = new_random_horizontal_translation_maximum;
}


void ImageDataSet::set_random_horizontal_translation_minimum(const type& new_random_horizontal_translation_minimum)
{
    random_horizontal_translation_minimum = new_random_horizontal_translation_minimum;
}


void ImageDataSet::set_random_vertical_translation_minimum(const type& new_random_vertical_translation_minimum)
{
    random_vertical_translation_minimum = new_random_vertical_translation_minimum;
}


void ImageDataSet::set_random_vertical_translation_maximum(const type& new_random_vertical_translation_maximum)
{
    random_vertical_translation_maximum = new_random_vertical_translation_maximum;
}


/// Serializes the data set object into a XML document of the TinyXML library without keep the DOM tree in memory.

void ImageDataSet::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    time_t start, finish;
    time(&start);

    file_stream.OpenElement("DataSet");

    // Data file

    file_stream.OpenElement("DataFile");

    // File type

    file_stream.OpenElement("FileType");
    file_stream.PushText("bmp");
    file_stream.CloseElement();

    // Data file name

    file_stream.OpenElement("DataSourcePath");
    file_stream.PushText(data_source_path.c_str());
    file_stream.CloseElement();

    // Rows labels

    file_stream.OpenElement("RowsLabels");
    buffer.str("");
    buffer << has_rows_labels;
    file_stream.PushText(buffer.str().c_str());
    file_stream.CloseElement();

    // Channels

    file_stream.OpenElement("Channels");
    buffer.str("");
    buffer << get_channels_number();
    file_stream.PushText(buffer.str().c_str());
    file_stream.CloseElement();

    // Width

    file_stream.OpenElement("Width");
    buffer.str("");
    buffer << get_image_width();
    file_stream.PushText(buffer.str().c_str());
    file_stream.CloseElement();

    // Height

    file_stream.OpenElement("Height");
    buffer.str("");
    buffer << get_image_height();
    file_stream.PushText(buffer.str().c_str());
    file_stream.CloseElement();

    // Padding

    file_stream.OpenElement("Padding");
    buffer.str("");
    buffer << get_image_padding();
    file_stream.PushText(buffer.str().c_str());
    file_stream.CloseElement();

    // Data augmentation

    file_stream.OpenElement("randomReflectionAxisX");
    buffer.str("");
    buffer << get_random_reflection_axis_x();
    file_stream.PushText(buffer.str().c_str());
    file_stream.CloseElement();

    file_stream.OpenElement("randomReflectionAxisY");
    buffer.str("");
    buffer << get_random_reflection_axis_y();
    file_stream.PushText(buffer.str().c_str());
    file_stream.CloseElement();

    file_stream.OpenElement("randomRotationMinimum");
    buffer.str("");
    buffer << get_random_rotation_minimum();
    file_stream.PushText(buffer.str().c_str());
    file_stream.CloseElement();

    file_stream.OpenElement("randomRotationMaximum");
    buffer.str("");
    buffer << get_random_rotation_maximum();
    file_stream.PushText(buffer.str().c_str());
    file_stream.CloseElement();

    file_stream.OpenElement("randomHorizontalTranslationMinimum");
    buffer.str("");
    buffer << get_random_horizontal_translation_minimum();
    file_stream.PushText(buffer.str().c_str());
    file_stream.CloseElement();

    file_stream.OpenElement("randomHorizontalTranslationMaximum");
    buffer.str("");
    buffer << get_random_horizontal_translation_maximum();
    file_stream.PushText(buffer.str().c_str());
    file_stream.CloseElement();

    file_stream.OpenElement("randomVerticalTranslationMinimum");
    buffer.str("");
    buffer << get_random_vertical_translation_minimum();
    file_stream.PushText(buffer.str().c_str());
    file_stream.CloseElement();

    file_stream.OpenElement("randomVerticalTranslationMaximum");
    buffer.str("");
    buffer << get_random_vertical_translation_maximum();
    file_stream.PushText(buffer.str().c_str());
    file_stream.CloseElement();

    // Codification

    file_stream.OpenElement("Codification");
    buffer.str("");
    buffer << get_codification_string();
    file_stream.PushText(buffer.str().c_str());
    file_stream.CloseElement();

    // Close DataFile

    file_stream.CloseElement();

    // raw_variables

    file_stream.OpenElement("raw_variables");

    // raw_variables number
    {
        file_stream.OpenElement("raw_variablesNumber");

        buffer.str("");
        buffer << get_raw_variables_number();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // raw_variables items

    const Index raw_variables_number = get_raw_variables_number();

    {
        for(Index i = 0; i < raw_variables_number; i++)
        {
            file_stream.OpenElement("Column");

            file_stream.PushAttribute("Item", to_string(i+1).c_str());

            raw_variables(i).write_XML(file_stream);

            file_stream.CloseElement();
        }
    }

    // Close raw_variables

    file_stream.CloseElement();

    // Rows labels

    if(has_rows_labels)
    {
        const Index rows_labels_number = rows_labels.size();

        file_stream.OpenElement("RowsLabels");

        buffer.str("");

        for(Index i = 0; i < rows_labels_number; i++)
        {
            buffer << rows_labels(i);

            if(i != rows_labels_number-1) buffer << ",";
        }

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Samples

    file_stream.OpenElement("Samples");

    // Samples number
    {
        file_stream.OpenElement("SamplesNumber");

        buffer.str("");
        buffer << get_samples_number();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Samples uses

    {
        file_stream.OpenElement("SamplesUses");

        buffer.str("");

        const Index samples_number = get_samples_number();

        for(Index i = 0; i < samples_number; i++)
        {
            SampleUse sample_use = samples_uses(i);

            buffer << Index(sample_use);

            if(i < (samples_number-1)) buffer << " ";
        }

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Close samples

    file_stream.CloseElement();

    // Close data set

    file_stream.CloseElement();

    time(&finish);
}


void ImageDataSet::from_XML(const tinyxml2::XMLDocument& data_set_document)
{
    ostringstream buffer;

    // Data set element

    const tinyxml2::XMLElement* data_set_element = data_set_document.FirstChildElement("DataSet");

    if(!data_set_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Data set element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    // Data file

    const tinyxml2::XMLElement* data_file_element = data_set_element->FirstChildElement("DataFile");

    if(!data_file_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Data file element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    // File type

    const tinyxml2::XMLElement* file_type_element = data_file_element->FirstChildElement("FileType");

    if(!file_type_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "FileType element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if(file_type_element->GetText())
    {
        const string new_file_type = file_type_element->GetText();

//        set_data_source_path(new_data_file_name);
    }

    // Data file name

    const tinyxml2::XMLElement* data_file_name_element = data_file_element->FirstChildElement("DataSourcePath");

    if(!data_file_name_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "DataSourcePath element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if(data_file_name_element->GetText())
    {
        const string new_data_file_name = data_file_name_element->GetText();

        set_data_source_path(new_data_file_name);
    }

    // Rows labels

    const tinyxml2::XMLElement* rows_label_element = data_file_element->FirstChildElement("RowsLabels");

    if(rows_label_element)
    {
        const string new_rows_label_string = rows_label_element->GetText();

        try
        {
            set_has_rows_label(new_rows_label_string == "1");
        }
        catch(const exception& e)
        {
            cerr << e.what() << endl;
        }
    }


    // Channels

    const tinyxml2::XMLElement* channels_number_element = data_file_element->FirstChildElement("Channels");

    if(channels_number_element)
    {
        if(channels_number_element->GetText())
        {
            const Index channels = Index(atoi(channels_number_element->GetText()));

            set_channels_number(channels);
        }
    }

    // Width

    const tinyxml2::XMLElement* image_width_element = data_file_element->FirstChildElement("Width");

    if(image_width_element)
    {
        if(image_width_element->GetText())
        {
            const Index width = Index(atoi(image_width_element->GetText()));

            set_image_width(width);
        }
    }

    // Height

    const tinyxml2::XMLElement* image_height_element = data_file_element->FirstChildElement("Height");

    if(image_height_element)
    {
        if(image_height_element->GetText())
        {
            const Index height = Index(atoi(image_height_element->GetText()));

            set_image_height(height);
        }
    }

    // Padding

    const tinyxml2::XMLElement* image_padding_element = data_file_element->FirstChildElement("Padding");

    if(image_padding_element)
    {
        if(image_padding_element->GetText())
        {
            const Index padding = Index(atoi(image_padding_element->GetText()));

            set_image_padding(padding);
        }
    }

    // Data augmentation

    const tinyxml2::XMLElement* random_reflection_axis_element_x = data_file_element->FirstChildElement("randomReflectionAxisX");

    if(random_reflection_axis_element_x)
    {
        if(random_reflection_axis_element_x->GetText())
        {
            const bool randomReflectionAxisX = Index(atoi(random_reflection_axis_element_x->GetText()));

            set_random_reflection_axis_x(randomReflectionAxisX);

        }
    }

    const tinyxml2::XMLElement* random_reflection_axis_element_y = data_file_element->FirstChildElement("randomReflectionAxisY");

    if(random_reflection_axis_element_x)
    {
        if(random_reflection_axis_element_x->GetText())
        {
            const bool randomReflectionAxisY = Index(atoi(random_reflection_axis_element_y->GetText()));

            set_random_reflection_axis_y(randomReflectionAxisY);

        }
    }

    const tinyxml2::XMLElement* random_rotation_minimum = data_file_element->FirstChildElement("randomRotationMinimum");

    if(random_rotation_minimum)
    {
        if(random_rotation_minimum->GetText())
        {
            const type randomRotationMinimum = type(atoi(random_rotation_minimum->GetText()));

            set_random_rotation_minimum(randomRotationMinimum);

        }
    }

    const tinyxml2::XMLElement* random_rotation_maximum = data_file_element->FirstChildElement("randomRotationMaximum");

    if(random_rotation_maximum)
    {
        if(random_rotation_maximum->GetText())
        {
            const type randomRotationMaximum = type(atoi(random_rotation_maximum->GetText()));

            set_random_rotation_minimum(randomRotationMaximum);

        }
    }

    const tinyxml2::XMLElement* random_horizontal_translation_minimum = data_file_element->FirstChildElement("randomHorizontalTranslationMinimum");

    if(random_horizontal_translation_minimum)
    {
        if(random_horizontal_translation_minimum->GetText())
        {
            const type randomHorizontalTranslationMinimum = type(atoi(random_horizontal_translation_minimum->GetText()));

            set_random_horizontal_translation_minimum(randomHorizontalTranslationMinimum);

        }
    }

    const tinyxml2::XMLElement* random_vertical_translation_minimum = data_file_element->FirstChildElement("randomVerticalTranslationMinimum");

    if(random_vertical_translation_minimum)
    {
        if(random_vertical_translation_minimum->GetText())
        {
            const type randomVerticalTranslationMinimum = type(atoi(random_vertical_translation_minimum->GetText()));

            set_random_vertical_translation_minimum(randomVerticalTranslationMinimum);

        }
    }

    const tinyxml2::XMLElement* random_horizontal_translation_maximum = data_file_element->FirstChildElement("randomHorizontalTranslationMaximum");

    if(random_horizontal_translation_maximum)
    {
        if(random_horizontal_translation_maximum->GetText())
        {
            const type randomHorizontalTranslationMaximum = type(atoi(random_horizontal_translation_maximum->GetText()));

            set_random_horizontal_translation_maximum(randomHorizontalTranslationMaximum);

        }
    }

    const tinyxml2::XMLElement* random_vertical_translation_maximum = data_file_element->FirstChildElement("randomVerticalTranslationMaximum");

    if(random_vertical_translation_maximum)
    {
        if(random_vertical_translation_maximum->GetText())
        {
            const type randomVerticalTranslationMaximum = type(atoi(random_vertical_translation_maximum->GetText()));

            set_random_vertical_translation_maximum(randomVerticalTranslationMaximum);

        }
    }

    const tinyxml2::XMLElement* codification_element = data_file_element->FirstChildElement("Codification");

    if(codification_element)
    {
        if(codification_element->GetText())
        {
            const string codification = codification_element->GetText();

            set_codification(codification);
        }
    }


    // raw_variables

    const tinyxml2::XMLElement* raw_variables_element = data_set_element->FirstChildElement("raw_variables");

    if(!raw_variables_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "raw_variables element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    // raw_variables number

    const tinyxml2::XMLElement* raw_variables_number_element = raw_variables_element->FirstChildElement("raw_variablesNumber");

    if(!raw_variables_number_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "raw_variables number element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    Index new_raw_variables_number = 0;

    if(raw_variables_number_element->GetText())
    {
        new_raw_variables_number = Index(atoi(raw_variables_number_element->GetText()));

        set_raw_variables_number(new_raw_variables_number);
    }

    // raw_variables

    const tinyxml2::XMLElement* start_element = raw_variables_number_element;

    if(new_raw_variables_number > 0)
    {
        for(Index i = 0; i < new_raw_variables_number; i++)
        {
            const tinyxml2::XMLElement* column_element = start_element->NextSiblingElement("Column");
            start_element = column_element;

            if(column_element->Attribute("Item") != to_string(i+1))
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void DataSet:from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Column item number (" << i+1 << ") does not match (" << column_element->Attribute("Item") << ").\n";

                throw runtime_error(buffer.str());
            }

            // Name

            const tinyxml2::XMLElement* name_element = column_element->FirstChildElement("Name");

            if(!name_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Name element is nullptr.\n";

                throw runtime_error(buffer.str());
            }

            if(name_element->GetText())
            {
                const string new_name = name_element->GetText();

                raw_variables(i).name = new_name;
            }

            // Scaler

            const tinyxml2::XMLElement* scaler_element = column_element->FirstChildElement("Scaler");

            if(!scaler_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void DataSet::from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Scaler element is nullptr.\n";

                throw runtime_error(buffer.str());
            }

            if(scaler_element->GetText())
            {
                const string new_scaler = scaler_element->GetText();

                raw_variables(i).set_scaler(new_scaler);
            }

            // Column use

            const tinyxml2::XMLElement* raw_variable_use_element = column_element->FirstChildElement("ColumnUse");

            if(!raw_variable_use_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void DataSet::from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Column use element is nullptr.\n";

                throw runtime_error(buffer.str());
            }

            if(raw_variable_use_element->GetText())
            {
                const string new_raw_variable_use = raw_variable_use_element->GetText();

                raw_variables(i).set_use(new_raw_variable_use);
            }

            // Type

            const tinyxml2::XMLElement* type_element = column_element->FirstChildElement("Type");

            if(!type_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Type element is nullptr.\n";

                throw runtime_error(buffer.str());
            }

            if(type_element->GetText())
            {
                const string new_type = type_element->GetText();
                raw_variables(i).set_type(new_type);
            }

            if(raw_variables(i).type == RawVariableType::Categorical || raw_variables(i).type == RawVariableType::Binary)
            {
                // Categories

                const tinyxml2::XMLElement* categories_element = column_element->FirstChildElement("Categories");

                if(!categories_element)
                {
                    buffer << "OpenNN Exception: DataSet class.\n"
                           << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
                           << "Categories element is nullptr.\n";

                    throw runtime_error(buffer.str());
                }

                if(categories_element->GetText())
                {
                    const string new_categories = categories_element->GetText();

                    raw_variables(i).categories = get_tokens(new_categories, ';');
                }

                // Categories uses

                const tinyxml2::XMLElement* categories_uses_element = column_element->FirstChildElement("CategoriesUses");

                if(!categories_uses_element)
                {
                    buffer << "OpenNN Exception: DataSet class.\n"
                           << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
                           << "Categories uses element is nullptr.\n";

                    throw runtime_error(buffer.str());
                }

                if(categories_uses_element->GetText())
                {
                    const string new_categories_uses = categories_uses_element->GetText();

                    raw_variables(i).set_categories_uses(get_tokens(new_categories_uses, ';'));
                }
            }
        }
    }

//    // Time series raw_variables

//    const tinyxml2::XMLElement* time_series_raw_variables_element = data_set_element->FirstChildElement("TimeSeriesraw_variables");

//    if(!time_series_raw_variables_element)
//    {
//        // do nothing
//    }
//    else
//    {
//        // Time series raw_variables number

//        const tinyxml2::XMLElement* time_series_raw_variables_number_element = time_series_raw_variables_element->FirstChildElement("TimeSeriesraw_variablesNumber");

//        if(!time_series_raw_variables_number_element)
//        {
//            buffer << "OpenNN Exception: DataSet class.\n"
//                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
//                   << "Time seires raw_variables number element is nullptr.\n";

//            throw runtime_error(buffer.str());
//        }

//        Index time_series_new_raw_variables_number = 0;

//        if(time_series_raw_variables_number_element->GetText())
//        {
//            time_series_new_raw_variables_number = Index(atoi(time_series_raw_variables_number_element->GetText()));

//            set_time_series_raw_variables_number(time_series_new_raw_variables_number);
//        }

//        // Time series raw_variables

//        const tinyxml2::XMLElement* time_series_start_element = time_series_raw_variables_number_element;

//        if(time_series_new_raw_variables_number > 0)
//        {
//            for(Index i = 0; i < time_series_new_raw_variables_number; i++)
//            {
//                const tinyxml2::XMLElement* time_series_raw_variable_element = time_series_start_element->NextSiblingElement("TimeSeriesColumn");
//                time_series_start_element = time_series_raw_variable_element;

//                if(time_series_raw_variable_element->Attribute("Item") != to_string(i+1))
//                {
//                    buffer << "OpenNN Exception: DataSet class.\n"
//                           << "void DataSet:from_XML(const tinyxml2::XMLDocument&) method.\n"
//                           << "Time series column item number (" << i+1 << ") does not match (" << time_series_raw_variable_element->Attribute("Item") << ").\n";

//                    throw runtime_error(buffer.str());
//                }

//                // Name

//                const tinyxml2::XMLElement* time_series_name_element = time_series_raw_variable_element->FirstChildElement("Name");

//                if(!time_series_name_element)
//                {
//                    buffer << "OpenNN Exception: DataSet class.\n"
//                           << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
//                           << "Time series name element is nullptr.\n";

//                    throw runtime_error(buffer.str());
//                }

//                if(time_series_name_element->GetText())
//                {
//                    const string time_series_new_name = time_series_name_element->GetText();

//                    time_series_raw_variables(i).name = time_series_new_name;
//                }

//                // Scaler

//                const tinyxml2::XMLElement* time_series_scaler_element = time_series_raw_variable_element->FirstChildElement("Scaler");

//                if(!time_series_scaler_element)
//                {
//                    buffer << "OpenNN Exception: DataSet class.\n"
//                           << "void DataSet::from_XML(const tinyxml2::XMLDocument&) method.\n"
//                           << "Time series scaler element is nullptr.\n";

//                    throw runtime_error(buffer.str());
//                }

//                if(time_series_scaler_element->GetText())
//                {
//                    const string time_series_new_scaler = time_series_scaler_element->GetText();

//                    time_series_raw_variables(i).set_scaler(time_series_new_scaler);
//                }

//                // Column use

//                const tinyxml2::XMLElement* time_series_raw_variable_use_element = time_series_raw_variable_element->FirstChildElement("ColumnUse");

//                if(!time_series_raw_variable_use_element)
//                {
//                    buffer << "OpenNN Exception: DataSet class.\n"
//                           << "void DataSet::from_XML(const tinyxml2::XMLDocument&) method.\n"
//                           << "Time series column use element is nullptr.\n";

//                    throw runtime_error(buffer.str());
//                }

//                if(time_series_raw_variable_use_element->GetText())
//                {
//                    const string time_series_new_raw_variable_use = time_series_raw_variable_use_element->GetText();

//                    time_series_raw_variables(i).set_use(time_series_new_raw_variable_use);
//                }

//                // Type

//                const tinyxml2::XMLElement* time_series_type_element = time_series_raw_variable_element->FirstChildElement("Type");

//                if(!time_series_type_element)
//                {
//                    buffer << "OpenNN Exception: DataSet class.\n"
//                           << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
//                           << "Time series type element is nullptr.\n";

//                    throw runtime_error(buffer.str());
//                }

//                if(time_series_type_element->GetText())
//                {
//                    const string time_series_new_type = time_series_type_element->GetText();
//                    time_series_raw_variables(i).set_type(time_series_new_type);
//                }

//                if(time_series_raw_variables(i).type == ColumnType::Categorical || time_series_raw_variables(i).type == ColumnType::Binary)
//                {
//                    // Categories

//                    const tinyxml2::XMLElement* time_series_categories_element = time_series_raw_variable_element->FirstChildElement("Categories");

//                    if(!time_series_categories_element)
//                    {
//                        buffer << "OpenNN Exception: DataSet class.\n"
//                               << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
//                               << "Time series categories element is nullptr.\n";

//                        throw runtime_error(buffer.str());
//                    }

//                    if(time_series_categories_element->GetText())
//                    {
//                        const string time_series_new_categories = time_series_categories_element->GetText();

//                        time_series_raw_variables(i).categories = get_tokens(time_series_new_categories, ';');
//                    }

//                    // Categories uses

//                    const tinyxml2::XMLElement* time_series_categories_uses_element = time_series_raw_variable_element->FirstChildElement("CategoriesUses");

//                    if(!time_series_categories_uses_element)
//                    {
//                        buffer << "OpenNN Exception: DataSet class.\n"
//                               << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
//                               << "Time series categories uses element is nullptr.\n";

//                        throw runtime_error(buffer.str());
//                    }

//                    if(time_series_categories_uses_element->GetText())
//                    {
//                        const string time_series_new_categories_uses = time_series_categories_uses_element->GetText();

//                        time_series_raw_variables(i).set_categories_uses(get_tokens(time_series_new_categories_uses, ';'));
//                    }
//                }
//            }
//        }
//    }

    // Rows label

    if(has_rows_labels)
    {
        // Rows labels begin tag

        const tinyxml2::XMLElement* rows_labels_element = data_set_element->FirstChildElement("RowsLabels");

        if(!rows_labels_element)
        {
            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Rows labels element is nullptr.\n";

            throw runtime_error(buffer.str());
        }

        // Rows labels

        if(rows_labels_element->GetText())
        {
            const string new_rows_labels = rows_labels_element->GetText();

            char separator = ',';

            if(new_rows_labels.find(",") == string::npos
                    && new_rows_labels.find(";") != string::npos) {
                separator = ';';
            }

            rows_labels = get_tokens(new_rows_labels, separator);
        }
    }

    // Samples

    const tinyxml2::XMLElement* samples_element = data_set_element->FirstChildElement("Samples");

    if(!samples_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Samples element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    // Samples number

    const tinyxml2::XMLElement* samples_number_element = samples_element->FirstChildElement("SamplesNumber");

    if(!samples_number_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Samples number element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if(samples_number_element->GetText())
    {
        const Index new_samples_number = Index(atoi(samples_number_element->GetText()));

        samples_uses.resize(new_samples_number);

        set_training();
    }

    // Samples uses

    const tinyxml2::XMLElement* samples_uses_element = samples_element->FirstChildElement("SamplesUses");

    if(!samples_uses_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Samples uses element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if(samples_uses_element->GetText())
    {
        set_samples_uses(get_tokens(samples_uses_element->GetText(), ' '));
    }

   // Missing values

   const tinyxml2::XMLElement* missing_values_element = data_set_element->FirstChildElement("MissingValues");

   if(!missing_values_element)
   {
       buffer << "OpenNN Exception: DataSet class.\n"
              << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
              << "Missing values element is nullptr.\n";

       throw runtime_error(buffer.str());
   }

   // Missing values method

   const tinyxml2::XMLElement* missing_values_method_element = missing_values_element->FirstChildElement("MissingValuesMethod");

   if(!missing_values_method_element)
   {
       buffer << "OpenNN Exception: DataSet class.\n"
              << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
              << "Missing values method element is nullptr.\n";

       throw runtime_error(buffer.str());
   }

   if(missing_values_method_element->GetText())
   {
       set_missing_values_method(missing_values_method_element->GetText());
   }

   // Missing values number

   const tinyxml2::XMLElement* missing_values_number_element = missing_values_element->FirstChildElement("MissingValuesNumber");

   if(!missing_values_number_element)
   {
       buffer << "OpenNN Exception: DataSet class.\n"
              << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
              << "Missing values number element is nullptr.\n";

       throw runtime_error(buffer.str());
   }

    if(missing_values_number_element->GetText())
    {
        missing_values_number = Index(atoi(missing_values_number_element->GetText()));
    }

   if(missing_values_number > 0)
   {
       // raw_variables Missing values number

       const tinyxml2::XMLElement* raw_variables_missing_values_number_element = missing_values_element->FirstChildElement("raw_variablesMissingValuesNumber");

       if(!raw_variables_missing_values_number_element)
       {
           buffer << "OpenNN Exception: DataSet class.\n"
                  << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                  << "raw_variables missing values number element is nullptr.\n";

           throw runtime_error(buffer.str());
       }

       if(raw_variables_missing_values_number_element->GetText())
       {
           Tensor<string, 1> new_raw_variables_missing_values_number = get_tokens(raw_variables_missing_values_number_element->GetText(), ' ');

           raw_variables_missing_values_number.resize(new_raw_variables_missing_values_number.size());

           for(Index i = 0; i < new_raw_variables_missing_values_number.size(); i++)
           {
               raw_variables_missing_values_number(i) = atoi(new_raw_variables_missing_values_number(i).c_str());
           }
       }

       // Rows missing values number

       const tinyxml2::XMLElement* rows_missing_values_number_element = missing_values_element->FirstChildElement("RowsMissingValuesNumber");

       if(!rows_missing_values_number_element)
       {
           buffer << "OpenNN Exception: DataSet class.\n"
                  << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                  << "Rows missing values number element is nullptr.\n";

           throw runtime_error(buffer.str());
       }

    if(missing_values_number_element->GetText())
        {
            rows_missing_values_number = Index(atoi(rows_missing_values_number_element->GetText()));
        }
    }

    // Display

    const tinyxml2::XMLElement* display_element = data_set_element->FirstChildElement("Display");

    if(display_element)
    {
        const string new_display_string = display_element->GetText();

        try
        {
            set_display(new_display_string != "0");
        }
        catch(const exception& e)
        {
            cerr << e.what() << endl;
        }
    }
}


void ImageDataSet::fill_image_data()
{
    if(data_source_path.empty())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void fill_image_data() method.\n"
               << "Data file name is empty.\n";

        throw runtime_error(buffer.str());
    }
/*
    has_raw_variables_names = true;
    has_rows_labels = true;

    images_number = images_path.size();
    const int classes_number = classes_folder.size();

    const int image_size = width * height * channels;

    if(classes_number == 2)
    {
        Index binary_raw_variables_number = 1;
        data.resize(images_number, image_size + binary_raw_variables_number);
//        imageDataAux.resize(images_number, image_size + binary_raw_variables_number);
    }
    else
    {
        data.resize(images_number, image_size + classes_number);
//        imageDataAux.resize(images_number, image_size + classes_number);
    }

//    memcpy(data.data(), imageDataAux.data(), images_number * image_size * sizeof(type));

    rows_labels.resize(images_number);

    Index row_index = 0;

    for(Index i = 0; i < classes_number; i++)
    {
        for(Index j = 0;  j < images_per_folder[i]; j++)
        {
            if(classes_number == 2 && i == 0)
            {
                data(row_index, image_size) = 1;
            }
            else if(classes_number == 2 && i == 1)
            {
                data(row_index, image_size) = type(0);
            }
            else
            {
                data(row_index, image_size + i) = 1;
            }

            rows_labels(row_index) = images_path[row_index];

            row_index++;
        }
    }

    raw_variables.resize(image_size + 1);

    // Input raw_variables

    Index raw_variable_index = 0;

    for(Index i = 0; i < channels; i++)
    {
        for(Index j = 0; j < width; j++)
        {
            for(Index k = 0; k < height ; k++)
            {
                raw_variables(raw_variable_index).name= "pixel_" + to_string(i+1)+ "_" + to_string(j+1) + "_" + to_string(k+1);
                raw_variables(raw_variable_index).type = RawVariableType::Numeric;
                raw_variables(raw_variable_index).raw_variable_use = VariableUse::Input;
                raw_variables(raw_variable_index).scaler = Scaler::MinimumMaximum;
                raw_variable_index++;
            }
        }
    }

    // Target raw_variables

    raw_variables(image_size).name = "class";

    if(classes_number == 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_bmp() method.\n"
               << "Invalid number of categories. The minimum is 2 and you have 1.\n";

        throw runtime_error(buffer.str());
    }

    Tensor<string, 1> categories(classes_number);

    for(Index i = 0 ; i < classes_number; i++)
    {
        categories(i) = classes_folder[i];
    }

    raw_variables(image_size).raw_variable_use = VariableUse::Target;
    raw_variables(image_size).categories = categories;

    raw_variables(image_size).categories_uses.resize(classes_number);
    raw_variables(image_size).categories_uses.setConstant(VariableUse::Target);

    if(classes_number == 2)
    {
        raw_variables(image_size).type = RawVariableType::Binary;
    }
    else
    {
        raw_variables(image_size).type = RawVariableType::Categorical;
    }

    samples_uses.resize(images_number);
    split_samples_random();

    image_width = width;
    image_height = height;
    channels_number = channels;

    input_variables_dimensions.resize(3);
    input_variables_dimensions.setValues({height, width, channels});
*/
}


void ImageDataSet::set_categories_number(const Index& new_categories_number)
{
    categories_number = new_categories_number;
}


Tensor<type, 2> ImageDataSet::read_bmp(const string& path)
{
    vector<fs::path> folderPath;
    vector<fs::path> imagePath;

    for (const auto &entry : fs::directory_iterator(path))
    {
        if (fs::is_directory(entry))
        {
          folderPath.emplace_back(entry.path().string());
        }
    }

    Index classes_number = folderPath.size();
    Tensor<Index, 1> image_number(classes_number+1);
    Index samples_number = 0;

    image_number.setZero();

    for (Index i = 0; i < classes_number; i++)
    {
        for (const auto &entry : fs::directory_iterator(folderPath[i]))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".bmp")
            {
                imagePath.emplace_back(entry.path().string());
                samples_number++;
            }
        }

        image_number[i+1] = samples_number;
    }

    ImageData imgdata;// = read_bmp_image_gpt(imagePath[0]);
    Index pixels_number = imgdata.channels * imgdata.width * imgdata.height;
    Tensor<type, 2> data(samples_number, pixels_number + 1);

    if (classes_number > 2)
    {
        data.resize(samples_number, pixels_number + classes_number);
    }

    data.setZero();

    for (Index i = 0; i < samples_number; i++)
    {
        ImageData imgdata;// = read_bmp_image_gpt(imagePath[i]);

        if(pixels_number != imgdata.channels * imgdata.width * imgdata.height)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class.\n"
                   << "Tensor<type, 2> read_bmp() method.\n"
                   << "Some images of the dataset have different channel number, width and/or height.\n";

            throw invalid_argument(buffer.str());
        }

        for (Index j = 0; j < pixels_number; j++)
        {
            data(i,j) = imgdata.data[j];
        }

        if (classes_number == 2)
        {
            if (i >= image_number[0] && i < image_number[1])
            {
                data(i,pixels_number) = 1;
            }
        }
        else
        {
            for (Index k = 0; k < classes_number; k++)
            {
                if (i >= image_number[k] && i < image_number[k+1])
                {
                    data(i,k+pixels_number) = 1;
                    break;
                }
            }
        }
    }

    return data;
}


} // opennn namespace


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
