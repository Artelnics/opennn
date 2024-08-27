//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I M A G E   D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "image_data_set.h"
#include "images.h"
#include "tensors.h"
#include "strings_utilities.h"
#include "filesystem"

using namespace std::filesystem;

namespace opennn
{


ImageDataSet::ImageDataSet() : DataSet()
{

}


ImageDataSet::ImageDataSet(const Index& new_classes_number,
                           const Index& new_height,
                           const Index& new_width,
                           const Index& new_channels,
                           const Index& new_targets_number)
{
    set(new_classes_number, new_height, new_width, new_channels, new_targets_number);
}


Index ImageDataSet::get_channels_number() const
{
    return input_dimensions[2];
}


Index ImageDataSet::get_image_width() const
{
    return input_dimensions[1];
}


Index ImageDataSet::get_image_height() const
{
    return input_dimensions[0];
}


Index ImageDataSet::get_image_size() const
{
    return input_dimensions[0] * input_dimensions[1] * input_dimensions[2];
}


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
                       const Index& new_channels,
                       const Index& new_targets_number)
{
    model_type = ModelType::ImageClassification;

    const Index inputs_number = new_height * new_width * new_channels;
    const Index raw_variables_number = inputs_number + 1;
    const Index variables_number = inputs_number + new_targets_number;

    // Dimensions

    input_dimensions.resize(3);
    input_dimensions = { new_height, new_width, new_channels };

    target_dimensions.resize(1);
    target_dimensions = { new_targets_number };

    // Data

    data.resize(new_images_number, variables_number);

    // Variables

    raw_variables.resize(raw_variables_number);

    for(Index i = 0; i < inputs_number; i++)
    {
        raw_variables(i).name = "p_" + to_string(i+1);
        raw_variables(i).use = VariableUse::Input;
        raw_variables(i).type = RawVariableType::Numeric;
        raw_variables(i).scaler = Scaler::ImageMinMax;
    }

    if(new_targets_number == 1)
    {
        Tensor<string, 1> categories(new_targets_number);
        categories.setConstant("ABC");

        raw_variables(raw_variables_number-1).type = RawVariableType::Binary;
        raw_variables(raw_variables_number-1).use = VariableUse::Target;
        raw_variables(raw_variables_number-1).name = "target";

        raw_variables(raw_variables_number-1).set_categories(categories);
    }
    else
    {
        Tensor<string, 1> categories(new_targets_number);
        categories.setConstant("ABC");

        raw_variables(raw_variables_number-1).type = RawVariableType::Categorical;
        raw_variables(raw_variables_number-1).use = VariableUse::Target;
        raw_variables(raw_variables_number-1).name = "target";

        raw_variables(raw_variables_number-1).set_categories(categories);
    }

    // Samples

    samples_uses.resize(new_images_number);
    split_samples_random();

    set_raw_variables_scalers(Scaler::ImageMinMax);
}


void ImageDataSet::set_channels_number(const int& new_channels)
{
    input_dimensions[2] = new_channels;
}


void ImageDataSet::set_image_width(const int& new_width)
{
    input_dimensions[1] = new_width;
}


void ImageDataSet::set_image_height(const int& new_height)
{
    input_dimensions[0] = new_height;
}


void ImageDataSet::set_image_padding(const int& new_padding)
{
    padding = new_padding;
}


// void ImageDataSet::set_images_number(const Index & new_classes_number)
// {
//     images_number = new_classes_number;
// }


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


void ImageDataSet::to_XML(tinyxml2::XMLPrinter& file_stream) const
{

    file_stream.OpenElement("ImageDataSet");

    // Data file

    file_stream.OpenElement("DataSource");

    // File type

    file_stream.OpenElement("FileType");
    file_stream.PushText("bmp");
    file_stream.CloseElement();

    // Data file name

    file_stream.OpenElement("DataSourcePath");
    file_stream.PushText(data_source_path.c_str());
    file_stream.CloseElement();

    // Rows labels

    file_stream.OpenElement("HasIds");
    file_stream.PushText(to_string(has_ids).c_str());
    file_stream.CloseElement();

    // Channels

    file_stream.OpenElement("Channels");
    file_stream.PushText(to_string(get_channels_number()).c_str());
    file_stream.CloseElement();

    // Width

    file_stream.OpenElement("Width");
    file_stream.PushText(to_string(get_image_width()).c_str());
    file_stream.CloseElement();

    // Height

    file_stream.OpenElement("Height");
    file_stream.PushText(to_string(get_image_height()).c_str());
    file_stream.CloseElement();

    // Padding

    file_stream.OpenElement("Padding");
    file_stream.PushText(to_string(get_image_padding()).c_str());
    file_stream.CloseElement();

    // Data augmentation

    file_stream.OpenElement("RandomReflectionAxisX");
    file_stream.PushText(to_string(get_random_reflection_axis_x()).c_str());
    file_stream.CloseElement();

    file_stream.OpenElement("RandomReflectionAxisY");
    file_stream.PushText(to_string(get_random_reflection_axis_y()).c_str());
    file_stream.CloseElement();

    file_stream.OpenElement("RandomRotationMinimum");
    file_stream.PushText(to_string(get_random_rotation_minimum()).c_str());
    file_stream.CloseElement();

    file_stream.OpenElement("RandomRotationMaximum");
    file_stream.PushText(to_string(get_random_rotation_maximum()).c_str());
    file_stream.CloseElement();

    file_stream.OpenElement("RandomHorizontalTranslationMinimum");
    file_stream.PushText(to_string(get_random_horizontal_translation_minimum()).c_str());
    file_stream.CloseElement();

    file_stream.OpenElement("RandomHorizontalTranslationMaximum");
    file_stream.PushText(to_string(get_random_horizontal_translation_maximum()).c_str());
    file_stream.CloseElement();

    file_stream.OpenElement("RandomVerticalTranslationMinimum");
    file_stream.PushText(to_string(get_random_vertical_translation_minimum()).c_str());
    file_stream.CloseElement();

    file_stream.OpenElement("RandomVerticalTranslationMaximum");
    file_stream.PushText(to_string(get_random_vertical_translation_maximum()).c_str());
    file_stream.CloseElement();

    // Codification

    file_stream.OpenElement("Codification");
    file_stream.PushText(get_codification_string().c_str());
    file_stream.CloseElement();

    // Close DataFile

    file_stream.CloseElement();

    // Raw variables

    file_stream.OpenElement("RawVariables");

    // Raw variables number

    file_stream.OpenElement("RawVariablesNumber");
    file_stream.PushText(to_string(get_raw_variables_number()).c_str());
    file_stream.CloseElement();

    // Raw variables items

    const Index raw_variables_number = get_raw_variables_number();

    for(Index i = 0; i < raw_variables_number; i++)
    {
        file_stream.OpenElement("RawVariable");
        file_stream.PushAttribute("Item", to_string(i+1).c_str());
        raw_variables(i).to_XML(file_stream);
        file_stream.CloseElement();
    }

    // Close raw_variables

    file_stream.CloseElement();

    // Rows labels

    if(has_ids)
    {
        file_stream.OpenElement("Ids");
        file_stream.PushText(string_tensor_to_string(ids).c_str());

        file_stream.CloseElement();
    }

    // Samples

    file_stream.OpenElement("Samples");

    // Samples number

    file_stream.OpenElement("SamplesNumber");
    file_stream.PushText(to_string(get_samples_number()).c_str());
    file_stream.CloseElement();

    // Samples uses

    file_stream.OpenElement("SamplesUses");
    file_stream.PushText(tensor_to_string(get_samples_uses_tensor()).c_str());
    file_stream.CloseElement();

    // Close samples

    file_stream.CloseElement();

    // Close data set

    file_stream.CloseElement();
}


void ImageDataSet::from_XML(const tinyxml2::XMLDocument& data_set_document)
{
    ostringstream buffer;

    // Data set element

    const tinyxml2::XMLElement* image_data_set_element = data_set_document.FirstChildElement("ImageDataSet");

    if(!image_data_set_element)
        throw runtime_error("Data set element is nullptr.\n");

    // Data file

    const tinyxml2::XMLElement* data_source_element = image_data_set_element->FirstChildElement("DataSource");

    if(!data_source_element)
        throw runtime_error("Data source element is nullptr.\n");

    // File type

    const tinyxml2::XMLElement* file_type_element = data_source_element->FirstChildElement("FileType");

    if(!file_type_element)
        throw runtime_error("FileType element is nullptr.\n");

    if(file_type_element->GetText())
        set_data_source_path(file_type_element->GetText());

    // Data file name

    const tinyxml2::XMLElement* data_source_path_element = data_source_element->FirstChildElement("DataSourcePath");

    if(!data_source_path_element)
        throw runtime_error("DataSourcePath element is nullptr.\n");

    if(data_source_path_element->GetText())
        set_data_source_path(data_source_path_element->GetText());

    // Rows labels

    const tinyxml2::XMLElement* rows_label_element = data_source_element->FirstChildElement("HasIds");

    if(rows_label_element)
    {
        const string new_rows_label_string = rows_label_element->GetText();

        try
        {
            set_has_ids(new_rows_label_string == "1");
        }
        catch(const exception& e)
        {
            cerr << e.what() << endl;
        }
    }

    // Channels

    const tinyxml2::XMLElement* channels_number_element = data_source_element->FirstChildElement("Channels");

    if(channels_number_element)
        if(channels_number_element->GetText())
            set_channels_number(Index(atoi(channels_number_element->GetText())));

    // Width

    const tinyxml2::XMLElement* image_width_element = data_source_element->FirstChildElement("Width");

    if(image_width_element)
        if(image_width_element->GetText())
            set_image_width(Index(atoi(image_width_element->GetText())));

    // Height

    const tinyxml2::XMLElement* image_height_element = data_source_element->FirstChildElement("Height");

    if(image_height_element)
        if(image_height_element->GetText())
            set_image_height(Index(atoi(image_height_element->GetText())));

    // Padding

    const tinyxml2::XMLElement* image_padding_element = data_source_element->FirstChildElement("Padding");

    if(image_padding_element)
        if(image_padding_element->GetText())
            set_image_padding(Index(atoi(image_padding_element->GetText())));

    // Data augmentation

    const tinyxml2::XMLElement* random_reflection_axis_element_x = data_source_element->FirstChildElement("RandomReflectionAxisX");

    if(random_reflection_axis_element_x)
        if(random_reflection_axis_element_x->GetText())
            set_random_reflection_axis_x(Index(atoi(random_reflection_axis_element_x->GetText())));

    const tinyxml2::XMLElement* random_reflection_axis_element_y = data_source_element->FirstChildElement("RandomReflectionAxisY");

    if(random_reflection_axis_element_x)
        if(random_reflection_axis_element_x->GetText())
            set_random_reflection_axis_y(Index(atoi(random_reflection_axis_element_y->GetText())));

    const tinyxml2::XMLElement* random_rotation_minimum = data_source_element->FirstChildElement("RandomRotationMinimum");

    if(random_rotation_minimum)
        if(random_rotation_minimum->GetText())
            set_random_rotation_minimum(type(atoi(random_rotation_minimum->GetText())));

    const tinyxml2::XMLElement* random_rotation_maximum = data_source_element->FirstChildElement("RandomRotationMaximum");

    if(random_rotation_maximum)
        if(random_rotation_maximum->GetText())
            set_random_rotation_minimum(type(atoi(random_rotation_maximum->GetText())));

    const tinyxml2::XMLElement* random_horizontal_translation_minimum = data_source_element->FirstChildElement("RandomHorizontalTranslationMinimum");

    if(random_horizontal_translation_minimum)
        if(random_horizontal_translation_minimum->GetText())
            set_random_horizontal_translation_minimum(type(atoi(random_horizontal_translation_minimum->GetText())));

    const tinyxml2::XMLElement* random_vertical_translation_minimum = data_source_element->FirstChildElement("RandomVerticalTranslationMinimum");

    if(random_vertical_translation_minimum)
        if(random_vertical_translation_minimum->GetText())
            set_random_vertical_translation_minimum(type(atoi(random_vertical_translation_minimum->GetText())));

    const tinyxml2::XMLElement* random_horizontal_translation_maximum = data_source_element->FirstChildElement("RandomHorizontalTranslationMaximum");

    if(random_horizontal_translation_maximum)
        if(random_horizontal_translation_maximum->GetText())
            set_random_horizontal_translation_maximum(type(atoi(random_horizontal_translation_maximum->GetText())));

    const tinyxml2::XMLElement* random_vertical_translation_maximum = data_source_element->FirstChildElement("RandomVerticalTranslationMaximum");

    if(random_vertical_translation_maximum)
        if(random_vertical_translation_maximum->GetText())
            set_random_vertical_translation_maximum(type(atoi(random_vertical_translation_maximum->GetText())));

    const tinyxml2::XMLElement* codification_element = data_source_element->FirstChildElement("Codification");

    if(codification_element)
        if(codification_element->GetText())
            set_codification(codification_element->GetText());

    // RawVariables

    const tinyxml2::XMLElement* raw_variables_element = image_data_set_element->FirstChildElement("RawVariables");

    if(!raw_variables_element)
        throw runtime_error("RawVariables element is nullptr.\n");

    // Raw variables number

    const tinyxml2::XMLElement* raw_variables_number_element = raw_variables_element->FirstChildElement("RawVariablesNumber");

    if(!raw_variables_number_element)
        throw runtime_error("RawVariablesNumber element is nullptr.\n");

    Index new_raw_variables_number = 0;

    if(raw_variables_number_element->GetText())
    {
        new_raw_variables_number = Index(atoi(raw_variables_number_element->GetText()));

        set_raw_variables_number(new_raw_variables_number);
    }

    // Raw variables

    const tinyxml2::XMLElement* start_element = raw_variables_number_element;

    for(Index i = 0; i < new_raw_variables_number; i++)
    {
        const tinyxml2::XMLElement* raw_variable_element = start_element->NextSiblingElement("RawVariable");
        start_element = raw_variable_element;

        if(raw_variable_element->Attribute("Item") != to_string(i+1))
        {
            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void DataSet:from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "raw_variable item number (" << i+1 << ") does not match (" << raw_variable_element->Attribute("Item") << ").\n";

            throw runtime_error(buffer.str());
        }

        // Name

        const tinyxml2::XMLElement* name_element = raw_variable_element->FirstChildElement("Name");

        if(!name_element)
        {
            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void raw_variable::from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Name element is nullptr.\n";

            throw runtime_error(buffer.str());
        }

        if(name_element->GetText())
        {
            const string new_name = name_element->GetText();

            raw_variables(i).name = new_name;
        }

        // Scaler

        const tinyxml2::XMLElement* scaler_element = raw_variable_element->FirstChildElement("Scaler");

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

        // raw_variable use

        const tinyxml2::XMLElement* use_element = raw_variable_element->FirstChildElement("Use");

        if(!use_element)
        {
            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void DataSet::from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "raw_variable use element is nullptr.\n";

            throw runtime_error(buffer.str());
        }

        if(use_element->GetText())
        {
            const string new_raw_variable_use = use_element->GetText();

            raw_variables(i).set_use(new_raw_variable_use);
        }

        // Type

        const tinyxml2::XMLElement* type_element = raw_variable_element->FirstChildElement("Type");

        if(!type_element)
        {
            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void raw_variable::from_XML(const tinyxml2::XMLDocument&) method.\n"
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

            const tinyxml2::XMLElement* categories_element = raw_variable_element->FirstChildElement("Categories");

            if(!categories_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void raw_variable::from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Categories element is nullptr.\n";

                throw runtime_error(buffer.str());
            }

            if(categories_element->GetText())
            {
                const string new_categories = categories_element->GetText();

                raw_variables(i).categories = get_tokens(new_categories, ";");

            }
        }
    }

//    // Time series raw_variables

//    const tinyxml2::XMLElement* time_series_raw_variables_element = data_set_element->FirstChildElement("TimeSeriesRawVariables");

//    if(!time_series_raw_variables_element)
//    {
//        // do nothing
//    }
//    else
//    {
//        // Time series raw_variables number

//        const tinyxml2::XMLElement* time_series_raw_variables_number_element = time_series_raw_variables_element->FirstChildElement("TimeSeriesRawVariablesNumber");

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
//                const tinyxml2::XMLElement* time_series_raw_variable_element = time_series_start_element->NextSiblingElement("TimeSeriesRawVariable");
//                time_series_start_element = time_series_raw_variable_element;

//                if(time_series_raw_variable_element->Attribute("Item") != to_string(i+1))
//                {
//                    buffer << "OpenNN Exception: DataSet class.\n"
//                           << "void DataSet:from_XML(const tinyxml2::XMLDocument&) method.\n"
//                           << "Time series raw_variable item number (" << i+1 << ") does not match (" << time_series_raw_variable_element->Attribute("Item") << ").\n";

//                    throw runtime_error(buffer.str());
//                }

//                // Name

//                const tinyxml2::XMLElement* time_series_name_element = time_series_raw_variable_element->FirstChildElement("Name");

//                if(!time_series_name_element)
//                {
//                    buffer << "OpenNN Exception: DataSet class.\n"
//                           << "void raw_variable::from_XML(const tinyxml2::XMLDocument&) method.\n"
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

//                // raw_variable use

//                const tinyxml2::XMLElement* time_series_raw_variable_use_element = time_series_raw_variable_element->FirstChildElement("Use");

//                if(!time_series_raw_variable_use_element)
//                {
//                    buffer << "OpenNN Exception: DataSet class.\n"
//                           << "void DataSet::from_XML(const tinyxml2::XMLDocument&) method.\n"
//                           << "Time series raw_variable use element is nullptr.\n";

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
//                           << "void raw_variable::from_XML(const tinyxml2::XMLDocument&) method.\n"
//                           << "Time series type element is nullptr.\n";

//                    throw runtime_error(buffer.str());
//                }

//                if(time_series_type_element->GetText())
//                {
//                    const string time_series_new_type = time_series_type_element->GetText();
//                    time_series_raw_variables(i).set_type(time_series_new_type);
//                }

//                if(time_series_raw_variables(i).type == RawVariableType::Categorical || time_series_raw_variables(i).type == RawVariableType::Binary)
//                {
//                    // Categories

//                    const tinyxml2::XMLElement* time_series_categories_element = time_series_raw_variable_element->FirstChildElement("Categories");

//                    if(!time_series_categories_element)
//                    {
//                        buffer << "OpenNN Exception: DataSet class.\n"
//                               << "void raw_variable::from_XML(const tinyxml2::XMLDocument&) method.\n"
//                               << "Time series categories element is nullptr.\n";

//                        throw runtime_error(buffer.str());
//                    }

//                    if(time_series_categories_element->GetText())
//                    {
//                        const string time_series_new_categories = time_series_categories_element->GetText();

//                        time_series_raw_variables(i).categories = get_tokens(time_series_new_categories, ";);
//                    }
//                }
//            }
//        }
//    }

    // Rows label

    if(has_ids)
    {
        // Rows labels begin tag

        const tinyxml2::XMLElement* has_ids_element = image_data_set_element->FirstChildElement("HasIds");

        if(!has_ids_element)
        {
            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Rows labels element is nullptr.\n";

            throw runtime_error(buffer.str());
        }

        // Rows labels

        if(has_ids_element->GetText())
        {
            const string new_rows_labels = has_ids_element->GetText();

            string separator = ",";

            if(new_rows_labels.find(",") == string::npos
                    && new_rows_labels.find(";") != string::npos) {
                separator = ';';
            }

            ids = get_tokens(new_rows_labels, separator);

        }
    }

    // Samples

    const tinyxml2::XMLElement* samples_element = image_data_set_element->FirstChildElement("Samples");

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
        set_samples_uses(get_tokens(samples_uses_element->GetText(), " "));
    }

    // Display

    const tinyxml2::XMLElement* display_element = image_data_set_element->FirstChildElement("Display");

    if(display_element)
        set_display(display_element->GetText() != string("0"));
}


void ImageDataSet::read_bmp()
{
    vector<path> directory_path;
    vector<path> image_path;

    for(const directory_entry& current_directory : directory_iterator(data_source_path))
    {
        if(is_directory(current_directory))
        {
            directory_path.emplace_back(current_directory.path().string());
        }
    }

    const Index folders_number = directory_path.size();

    Tensor<Index, 1> images_number(folders_number + 1);
    images_number.setZero();

    Index samples_number = 0;

    for(Index i = 0; i < folders_number; i++)
    {
        for(const directory_entry& current_directory : directory_iterator(directory_path[i]))
        {
            if(current_directory.is_regular_file() && current_directory.path().extension() == ".bmp")
            {
                image_path.emplace_back(current_directory.path().string());
                samples_number++;
            }
        }

        images_number[i+1] = samples_number;
    }

    const Tensor<unsigned char, 3> image_data = read_bmp_image(image_path[0].string());

    const Index height = image_data.dimension(0);
    const Index width = image_data.dimension(1);
    const Index image_channels = image_data.dimension(2);

    const Index inputs_number = height * width * image_channels;
    const Index raw_variables_number = inputs_number + 1;

    const Index pixels_number = height * width * image_channels;

    const Index targets_number = (folders_number == 2) ? folders_number -1 : folders_number;

    set(samples_number, height, width, image_channels, targets_number);

    Tensor<string, 1> categories(targets_number);

    for(Index i = 0; i < targets_number; i++)
    {
        categories(i) = directory_path[i].filename().string();
    }

    raw_variables(raw_variables_number-1).set_categories(categories);

    data.setZero();

    #pragma omp parallel for
    for(Index i = 0; i < samples_number; i++)
    {
        const Tensor<unsigned char, 3> image_data = read_bmp_image(image_path[i].string());

        if(pixels_number != image_data.size())
            throw runtime_error("Different image sizes.\n");

        for(Index j = 0; j < pixels_number; j++)
        {
            data(i, j) = image_data(j);
        }

        if (targets_number == 1)
        {
            if (i >= images_number[0] && i < images_number[1])
            {
                data(i, pixels_number) = 0;
            }
            else
            {
                data(i, pixels_number) = 1;
            }
        }
        else
        {
            for (Index k = 0; k < targets_number; k++)
            {
                if (i >= images_number[k] && i < images_number[k + 1])
                {
                    data(i, k + pixels_number) = 1;
                    break;
                }
            }
        }

        if(display)
        {
            if(i % 1000 == 0)
                display_progress_bar(i, samples_number - 1000);
        }
    }

    if(display)
        cout << endl << "Image data set loaded." << endl;
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
