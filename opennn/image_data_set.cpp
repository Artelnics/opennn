//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I M A G E   D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "image_data_set.h"
#include "opennn_images.h"
#include "opennn_strings.h"

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

    data_source_path = "";

    const Index new_inputs_number = new_channels_number * new_width * new_height;

    const Index new_variables_number = new_channels_number * new_width * new_height + new_targets_number;

    data.resize(new_images_number, new_variables_number);

    columns.resize(new_variables_number);

    for(Index i = 0; i < new_variables_number; i++)
    {
        if(i < new_inputs_number)
        {
            columns(i).name = "column_" + to_string(i+1);
            columns(i).column_use = VariableUse::Input;
            columns(i).type = RawVariableType::Numeric;
        }
        else
        {
            columns(i).name = "column_" + to_string(i+1);
            columns(i).column_use = VariableUse::Target;
            columns(i).type = RawVariableType::Numeric;
        }
    }

    input_variables_dimensions.resize(3);
    input_variables_dimensions.setValues({new_height, new_width, new_channels_number});

    samples_uses.resize(new_images_number);
    split_samples_random();
}

void ImageDataSet::set_image_data_source_path(const string& new_data_source_path)
{
    image_data_source_path = new_data_source_path;
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


type ImageDataSet::calculate_intersection_over_union(const BoundingBox& bounding_box_1, const BoundingBox& bounding_box_2)
{
    const Index intersection_x_top_left = max(bounding_box_1.x_top_left, bounding_box_2.x_top_left);

    const Index intersection_y_top_left = max(bounding_box_1.y_top_left, bounding_box_2.y_top_left);

    const Index intersection_x_bottom_right = min(bounding_box_1.x_bottom_right, bounding_box_2.x_bottom_right);

    const Index intersection_y_bottom_right = min(bounding_box_1.y_bottom_right, bounding_box_2.y_bottom_right);

    if(intersection_x_bottom_right < intersection_x_top_left || intersection_y_bottom_right < intersection_y_top_left)
        return type(0);

    const type intersection_area = type((intersection_x_bottom_right - intersection_x_top_left)
                                                     * (intersection_y_bottom_right - intersection_y_top_left));

    const type bounding_box_1_area
        = type((bounding_box_1.x_bottom_right - bounding_box_1.x_top_left)*(bounding_box_1.y_bottom_right - bounding_box_1.y_top_left));

    const type bounding_box_2_area
        = type((bounding_box_2.x_bottom_right - bounding_box_2.x_top_left)*(bounding_box_2.y_bottom_right - bounding_box_2.y_top_left));

    const type union_area = bounding_box_1_area + bounding_box_2_area - intersection_area;

    const type intersection_over_union = type(intersection_area / union_area);

    return intersection_over_union;
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

    file_stream.OpenElement("DataFileName");
    file_stream.PushText(image_data_source_path.c_str());
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

    // Columns

    file_stream.OpenElement("Columns");

    // Columns number
    {
        file_stream.OpenElement("ColumnsNumber");

        buffer.str("");
        buffer << get_columns_number();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Columns items

    const Index columns_number = get_columns_number();

    {
        for(Index i = 0; i < columns_number; i++)
        {
            file_stream.OpenElement("Column");

            file_stream.PushAttribute("Item", to_string(i+1).c_str());

            columns(i).write_XML(file_stream);

            file_stream.CloseElement();
        }
    }

    // Close columns

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

        throw invalid_argument(buffer.str());
    }

    // Data file

    const tinyxml2::XMLElement* data_file_element = data_set_element->FirstChildElement("DataFile");

    if(!data_file_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Data file element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    // File type

    const tinyxml2::XMLElement* file_type_element = data_file_element->FirstChildElement("FileType");

    if(!file_type_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "FileType element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    if(file_type_element->GetText())
    {
        const string new_file_type = file_type_element->GetText();

//        set_data_source_path(new_data_file_name);
    }

    // Data file name

    const tinyxml2::XMLElement* data_file_name_element = data_file_element->FirstChildElement("DataFileName");

    if(!data_file_name_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "DataFileName element is nullptr.\n";

        throw invalid_argument(buffer.str());
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
        catch(const invalid_argument& e)
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


    // Columns

    const tinyxml2::XMLElement* columns_element = data_set_element->FirstChildElement("Columns");

    if(!columns_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Columns element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    // Columns number

    const tinyxml2::XMLElement* columns_number_element = columns_element->FirstChildElement("ColumnsNumber");

    if(!columns_number_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Columns number element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    Index new_columns_number = 0;

    if(columns_number_element->GetText())
    {
        new_columns_number = Index(atoi(columns_number_element->GetText()));

        set_columns_number(new_columns_number);
    }

    // Columns

    const tinyxml2::XMLElement* start_element = columns_number_element;

    if(new_columns_number > 0)
    {
        for(Index i = 0; i < new_columns_number; i++)
        {
            const tinyxml2::XMLElement* column_element = start_element->NextSiblingElement("Column");
            start_element = column_element;

            if(column_element->Attribute("Item") != to_string(i+1))
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void DataSet:from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Column item number (" << i+1 << ") does not match (" << column_element->Attribute("Item") << ").\n";

                throw invalid_argument(buffer.str());
            }

            // Name

            const tinyxml2::XMLElement* name_element = column_element->FirstChildElement("Name");

            if(!name_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Name element is nullptr.\n";

                throw invalid_argument(buffer.str());
            }

            if(name_element->GetText())
            {
                const string new_name = name_element->GetText();

                columns(i).name = new_name;
            }

            // Scaler

            const tinyxml2::XMLElement* scaler_element = column_element->FirstChildElement("Scaler");

            if(!scaler_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void DataSet::from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Scaler element is nullptr.\n";

                throw invalid_argument(buffer.str());
            }

            if(scaler_element->GetText())
            {
                const string new_scaler = scaler_element->GetText();

                columns(i).set_scaler(new_scaler);
            }

            // Column use

            const tinyxml2::XMLElement* column_use_element = column_element->FirstChildElement("ColumnUse");

            if(!column_use_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void DataSet::from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Column use element is nullptr.\n";

                throw invalid_argument(buffer.str());
            }

            if(column_use_element->GetText())
            {
                const string new_column_use = column_use_element->GetText();

                columns(i).set_use(new_column_use);
            }

            // Type

            const tinyxml2::XMLElement* type_element = column_element->FirstChildElement("Type");

            if(!type_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Type element is nullptr.\n";

                throw invalid_argument(buffer.str());
            }

            if(type_element->GetText())
            {
                const string new_type = type_element->GetText();
                columns(i).set_type(new_type);
            }

            if(columns(i).type == RawVariableType::Categorical || columns(i).type == RawVariableType::Binary)
            {
                // Categories

                const tinyxml2::XMLElement* categories_element = column_element->FirstChildElement("Categories");

                if(!categories_element)
                {
                    buffer << "OpenNN Exception: DataSet class.\n"
                           << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
                           << "Categories element is nullptr.\n";

                    throw invalid_argument(buffer.str());
                }

                if(categories_element->GetText())
                {
                    const string new_categories = categories_element->GetText();

                    columns(i).categories = get_tokens(new_categories, ';');
                }

                // Categories uses

                const tinyxml2::XMLElement* categories_uses_element = column_element->FirstChildElement("CategoriesUses");

                if(!categories_uses_element)
                {
                    buffer << "OpenNN Exception: DataSet class.\n"
                           << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
                           << "Categories uses element is nullptr.\n";

                    throw invalid_argument(buffer.str());
                }

                if(categories_uses_element->GetText())
                {
                    const string new_categories_uses = categories_uses_element->GetText();

                    columns(i).set_categories_uses(get_tokens(new_categories_uses, ';'));
                }
            }
        }
    }

//    // Time series columns

//    const tinyxml2::XMLElement* time_series_columns_element = data_set_element->FirstChildElement("TimeSeriesColumns");

//    if(!time_series_columns_element)
//    {
//        // do nothing
//    }
//    else
//    {
//        // Time series columns number

//        const tinyxml2::XMLElement* time_series_columns_number_element = time_series_columns_element->FirstChildElement("TimeSeriesColumnsNumber");

//        if(!time_series_columns_number_element)
//        {
//            buffer << "OpenNN Exception: DataSet class.\n"
//                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
//                   << "Time seires columns number element is nullptr.\n";

//            throw invalid_argument(buffer.str());
//        }

//        Index time_series_new_columns_number = 0;

//        if(time_series_columns_number_element->GetText())
//        {
//            time_series_new_columns_number = Index(atoi(time_series_columns_number_element->GetText()));

//            set_time_series_columns_number(time_series_new_columns_number);
//        }

//        // Time series columns

//        const tinyxml2::XMLElement* time_series_start_element = time_series_columns_number_element;

//        if(time_series_new_columns_number > 0)
//        {
//            for(Index i = 0; i < time_series_new_columns_number; i++)
//            {
//                const tinyxml2::XMLElement* time_series_column_element = time_series_start_element->NextSiblingElement("TimeSeriesColumn");
//                time_series_start_element = time_series_column_element;

//                if(time_series_column_element->Attribute("Item") != to_string(i+1))
//                {
//                    buffer << "OpenNN Exception: DataSet class.\n"
//                           << "void DataSet:from_XML(const tinyxml2::XMLDocument&) method.\n"
//                           << "Time series column item number (" << i+1 << ") does not match (" << time_series_column_element->Attribute("Item") << ").\n";

//                    throw invalid_argument(buffer.str());
//                }

//                // Name

//                const tinyxml2::XMLElement* time_series_name_element = time_series_column_element->FirstChildElement("Name");

//                if(!time_series_name_element)
//                {
//                    buffer << "OpenNN Exception: DataSet class.\n"
//                           << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
//                           << "Time series name element is nullptr.\n";

//                    throw invalid_argument(buffer.str());
//                }

//                if(time_series_name_element->GetText())
//                {
//                    const string time_series_new_name = time_series_name_element->GetText();

//                    time_series_columns(i).name = time_series_new_name;
//                }

//                // Scaler

//                const tinyxml2::XMLElement* time_series_scaler_element = time_series_column_element->FirstChildElement("Scaler");

//                if(!time_series_scaler_element)
//                {
//                    buffer << "OpenNN Exception: DataSet class.\n"
//                           << "void DataSet::from_XML(const tinyxml2::XMLDocument&) method.\n"
//                           << "Time series scaler element is nullptr.\n";

//                    throw invalid_argument(buffer.str());
//                }

//                if(time_series_scaler_element->GetText())
//                {
//                    const string time_series_new_scaler = time_series_scaler_element->GetText();

//                    time_series_columns(i).set_scaler(time_series_new_scaler);
//                }

//                // Column use

//                const tinyxml2::XMLElement* time_series_column_use_element = time_series_column_element->FirstChildElement("ColumnUse");

//                if(!time_series_column_use_element)
//                {
//                    buffer << "OpenNN Exception: DataSet class.\n"
//                           << "void DataSet::from_XML(const tinyxml2::XMLDocument&) method.\n"
//                           << "Time series column use element is nullptr.\n";

//                    throw invalid_argument(buffer.str());
//                }

//                if(time_series_column_use_element->GetText())
//                {
//                    const string time_series_new_column_use = time_series_column_use_element->GetText();

//                    time_series_columns(i).set_use(time_series_new_column_use);
//                }

//                // Type

//                const tinyxml2::XMLElement* time_series_type_element = time_series_column_element->FirstChildElement("Type");

//                if(!time_series_type_element)
//                {
//                    buffer << "OpenNN Exception: DataSet class.\n"
//                           << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
//                           << "Time series type element is nullptr.\n";

//                    throw invalid_argument(buffer.str());
//                }

//                if(time_series_type_element->GetText())
//                {
//                    const string time_series_new_type = time_series_type_element->GetText();
//                    time_series_columns(i).set_type(time_series_new_type);
//                }

//                if(time_series_columns(i).type == ColumnType::Categorical || time_series_columns(i).type == ColumnType::Binary)
//                {
//                    // Categories

//                    const tinyxml2::XMLElement* time_series_categories_element = time_series_column_element->FirstChildElement("Categories");

//                    if(!time_series_categories_element)
//                    {
//                        buffer << "OpenNN Exception: DataSet class.\n"
//                               << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
//                               << "Time series categories element is nullptr.\n";

//                        throw invalid_argument(buffer.str());
//                    }

//                    if(time_series_categories_element->GetText())
//                    {
//                        const string time_series_new_categories = time_series_categories_element->GetText();

//                        time_series_columns(i).categories = get_tokens(time_series_new_categories, ';');
//                    }

//                    // Categories uses

//                    const tinyxml2::XMLElement* time_series_categories_uses_element = time_series_column_element->FirstChildElement("CategoriesUses");

//                    if(!time_series_categories_uses_element)
//                    {
//                        buffer << "OpenNN Exception: DataSet class.\n"
//                               << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
//                               << "Time series categories uses element is nullptr.\n";

//                        throw invalid_argument(buffer.str());
//                    }

//                    if(time_series_categories_uses_element->GetText())
//                    {
//                        const string time_series_new_categories_uses = time_series_categories_uses_element->GetText();

//                        time_series_columns(i).set_categories_uses(get_tokens(time_series_new_categories_uses, ';'));
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

            throw invalid_argument(buffer.str());
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

        throw invalid_argument(buffer.str());
    }

    // Samples number

    const tinyxml2::XMLElement* samples_number_element = samples_element->FirstChildElement("SamplesNumber");

    if(!samples_number_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Samples number element is nullptr.\n";

        throw invalid_argument(buffer.str());
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

        throw invalid_argument(buffer.str());
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

       throw invalid_argument(buffer.str());
   }

   // Missing values method

   const tinyxml2::XMLElement* missing_values_method_element = missing_values_element->FirstChildElement("MissingValuesMethod");

   if(!missing_values_method_element)
   {
       buffer << "OpenNN Exception: DataSet class.\n"
              << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
              << "Missing values method element is nullptr.\n";

       throw invalid_argument(buffer.str());
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

       throw invalid_argument(buffer.str());
   }

    if(missing_values_number_element->GetText())
    {
        missing_values_number = Index(atoi(missing_values_number_element->GetText()));
    }

   if(missing_values_number > 0)
   {
       // Columns Missing values number

       const tinyxml2::XMLElement* columns_missing_values_number_element = missing_values_element->FirstChildElement("ColumnsMissingValuesNumber");

       if(!columns_missing_values_number_element)
       {
           buffer << "OpenNN Exception: DataSet class.\n"
                  << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                  << "Columns missing values number element is nullptr.\n";

           throw invalid_argument(buffer.str());
       }

       if(columns_missing_values_number_element->GetText())
       {
           Tensor<string, 1> new_columns_missing_values_number = get_tokens(columns_missing_values_number_element->GetText(), ' ');

           columns_missing_values_number.resize(new_columns_missing_values_number.size());

           for(Index i = 0; i < new_columns_missing_values_number.size(); i++)
           {
               columns_missing_values_number(i) = atoi(new_columns_missing_values_number(i).c_str());
           }
       }

       // Rows missing values number

       const tinyxml2::XMLElement* rows_missing_values_number_element = missing_values_element->FirstChildElement("RowsMissingValuesNumber");

       if(!rows_missing_values_number_element)
       {
           buffer << "OpenNN Exception: DataSet class.\n"
                  << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                  << "Rows missing values number element is nullptr.\n";

           throw invalid_argument(buffer.str());
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
        catch(const invalid_argument& e)
        {
            cerr << e.what() << endl;
        }
    }
}

Tensor<unsigned char, 1> ImageDataSet::read_bmp_image(const string& filename)
{
    FILE* file = fopen(filename.data(), "rb");

    if(!file)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_bmp_image() method.\n"
               << "Couldn't open the file.\n";

        throw invalid_argument(buffer.str());
    }

    unsigned char info[54];
    fread(info, sizeof(unsigned char), 54, file);

    const Index width_no_padding = *(int*)&info[18];
    image_height = *(int*)&info[22];
    const Index bits_per_pixel = *(int*)&info[28];
    int channels;

    bits_per_pixel == 24 ? channels = 3 : channels = 1;

    channels_number = channels;

    padding = 0;

    image_width = width_no_padding;

    while((channels*image_width + padding)% 4 != 0)
        padding++;

    const size_t size = image_height*(channels_number*image_width + padding);

    Tensor<unsigned char, 1> image(size);
    image.setZero();

    int data_offset = *(int*)(&info[0x0A]);
    fseek(file, (long int)(data_offset - 54), SEEK_CUR);

    fread(image.data(), sizeof(unsigned char), size, file);
    fclose(file);

    if(channels_number == 3)
    {
        const int rows_number = static_cast<int>(get_image_height());
        const int columns_number = static_cast<int>(get_image_width());

        Tensor<unsigned char, 1> data_without_padding = remove_padding(image, rows_number, columns_number, padding);

        const Eigen::array<Eigen::Index, 3> dims_3D = {channels, rows_number, columns_number};
        const Eigen::array<Eigen::Index, 1> dims_1D = {rows_number*columns_number};

        Tensor<unsigned char,1> red_channel_flatted = data_without_padding.reshape(dims_3D).chip(2,0).reshape(dims_1D); // row_major
        Tensor<unsigned char,1> green_channel_flatted = data_without_padding.reshape(dims_3D).chip(1,0).reshape(dims_1D); // row_major
        Tensor<unsigned char,1> blue_channel_flatted = data_without_padding.reshape(dims_3D).chip(0,0).reshape(dims_1D); // row_major

        Tensor<unsigned char,1> red_channel_flatted_sorted(red_channel_flatted.size());
        Tensor<unsigned char,1> green_channel_flatted_sorted(green_channel_flatted.size());
        Tensor<unsigned char,1> blue_channel_flatted_sorted(blue_channel_flatted.size());

        red_channel_flatted_sorted.setZero();
        green_channel_flatted_sorted.setZero();
        blue_channel_flatted_sorted.setZero();

        sort_channel(red_channel_flatted, red_channel_flatted_sorted, columns_number);
        sort_channel(green_channel_flatted, green_channel_flatted_sorted, columns_number);
        sort_channel(blue_channel_flatted, blue_channel_flatted_sorted,columns_number);

        Tensor<unsigned char, 1> red_green_concatenation(red_channel_flatted_sorted.size() + green_channel_flatted_sorted.size());
        red_green_concatenation = red_channel_flatted_sorted.concatenate(green_channel_flatted_sorted,0); // To allow a double concatenation

        image = red_green_concatenation.concatenate(blue_channel_flatted_sorted, 0);
    }

    return image;
}


void ImageDataSet::read_bmp()
{/*
    const fs::path path = data_source_path;

    if(data_source_path.empty())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_bmp() method.\n"
               << "Data file name is empty.\n";

        throw invalid_argument(buffer.str());
    }

    has_columns_names = true;
    has_rows_labels = true;

    separator = Separator::None;

    vector<fs::path> folder_paths;
    vector<fs::path> image_paths;

    for(const auto & entry : fs::directory_iterator(path))
    {
        folder_paths.emplace_back(entry.path().string());
    }


    for(Index i = 0 ; i < folder_paths.size() ; i++)
    {
        for(const auto & entry : fs::directory_iterator(folder_paths[i]))
        {
            image_paths.emplace_back(entry.path().string());
        }
    }

    for(Index i = 0; i < image_paths.size(); i++)
    {
        if(image_paths[i].extension() != ".bmp")
        {
            fs::remove_all(image_paths[i]);

            //            ostringstream buffer;

            //            buffer << "OpenNN Exception: DataSet class.\n"
            //                   << "void read_bmp() method.\n"
            //                   << "Non-bmp data file format found and deleted. Try to run the program again.\n";

            //            throw invalid_argument(buffer.str());
        }
    }

    Index classes_number = number_of_elements_in_directory(path);
    Tensor<Index, 1> images_numbers(classes_number);

    for(Index i = 0; i < classes_number; i++)
    {
        images_number += number_of_elements_in_directory(folder_paths[i]);
    }

    string info_img;
    Tensor<unsigned char,1> image;
    Index image_size;
    Index size_comprobation = 0;

    for(Index i = 0; i < image_paths.size(); i++)
    {
        info_img = image_paths[i].string();
        image = read_bmp_image(info_img);
        image_size = image.size();
        size_comprobation += image_size;
    }

    if(image_size != size_comprobation/image_paths.size())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_bmp() method.\n"
               << "Some images of the dataset have different channel number, width and/or height.\n";

        throw invalid_argument(buffer.str());
    }

    FILE* f = fopen(info_img.data(), "rb");

    unsigned char info[54];

    fread(info, sizeof(unsigned char), 54, f);
    const int width = *(int*)&info[18];
    const int height = *(int*)&info[22];

    const int paddingAmount = (4 - (width) % 4) % 4;
    const int paddingWidth = width + paddingAmount;

    const int bits_per_pixel = *(int*)&info[28];
    int channels;

    bits_per_pixel == 24 ? channels = 3 : channels = 1;

    if(classes_number == 2)
    {
        Index binary_columns_number = 1;
        data.resize(images_number, image_size + binary_columns_number);
    }
    else
    {
        data.resize(images_number, image_size + classes_number);
    }

    data.setZero();

    rows_labels.resize(images_number);

    Index row_index = 0;

    for(Index i = 0; i < classes_number; i++)
    {
        Index images_number = 0;

        vector<string> images_paths;

        for(const auto & entry : fs::directory_iterator(folder_paths[i]))
        {
            images_paths.emplace_back(entry.path().string());
        }

        images_number = images_paths.size();

        for(Index j = 0;  j < images_number; j++)
        {
            image = read_bmp_image(images_paths[j]);

            for(Index k = 0; k < image_size; k++)
            {
                data(row_index, k) = type(image[k]);
            }

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

            rows_labels(row_index) = images_paths[j];

            row_index++;
        }
    }

    columns.resize(image_size + 1);

    // Input columns

    Index column_index = 0;

    for(Index i = 0; i < channels; i++)
    {
        for(Index j = 0; j < paddingWidth; j++)
        {
            for(Index k = 0; k < height ; k++)
            {
                columns(column_index).name= "pixel_" + to_string(i+1)+ "_" + to_string(j+1) + "_" + to_string(k+1);
                columns(column_index).type = ColumnType::Numeric;
                columns(column_index).column_use = VariableUse::Input;
                columns(column_index).scaler = Scaler::MinimumMaximum;
                column_index++;
            }
        }
    }

    // Target columns

    columns(image_size).name = "class";

    if(classes_number == 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_bmp() method.\n"
               << "Invalid number of categories. The minimum is 2 and you have 1.\n";

        throw invalid_argument(buffer.str());


    }else if(classes_number == 2)
    {
        Tensor<string, 1> categories(classes_number);

        for(Index i = 0 ; i < classes_number; i++)
        {
            categories(i) = folder_paths[i].filename().string();
        }

        columns(image_size).column_use = VariableUse::Target;
        columns(image_size).type = ColumnType::Binary;
        columns(image_size).categories = categories;

        columns(image_size).categories_uses.resize(classes_number);
        columns(image_size).categories_uses.setConstant(VariableUse::Target);
    }else
    {
        Tensor<string, 1> categories(classes_number);

        for(Index i = 0 ; i < classes_number ; i++)
        {
            categories(i) = folder_paths[i].filename().string();
        }

        columns(image_size).column_use = VariableUse::Target;
        columns(image_size).type = ColumnType::Categorical;
        columns(image_size).categories = categories;

        columns(image_size).categories_uses.resize(classes_number);
        columns(image_size).categories_uses.setConstant(VariableUse::Target);
    }

    samples_uses.resize(images_number);
    split_samples_random();

    image_width = paddingWidth;
    image_height = height;

    input_variables_dimensions.resize(3);
    input_variables_dimensions.setValues({channels, paddingWidth, height});*/
}

void ImageDataSet::fill_image_data(const string& new_data_source_path, const vector<string>& classes_folder,
                                   const vector<string>& images_path, const vector<int>& images_per_folder,
                                   const int& width, const int& height, const int& channels, const Tensor<type, 2>& imageData)
{
    if(new_data_source_path.empty())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void fill_image_data() method.\n"
               << "Data file name is empty.\n";

        throw invalid_argument(buffer.str());
    }

    has_columns_names = true;
    has_rows_labels = true;

    separator = Separator::None;

    images_number = images_path.size();
    const int classes_number = classes_folder.size();

    const int image_size = width * height * channels;

    Tensor<type, 2> imageDataAux(imageData);

    if(classes_number == 2)
    {
        Index binary_columns_number = 1;
        data.resize(images_number, image_size + binary_columns_number);
//        imageDataAux.resize(images_number, image_size + binary_columns_number);
    }
    else
    {
        data.resize(images_number, image_size + classes_number);
//        imageDataAux.resize(images_number, image_size + classes_number);
    }

    memcpy(data.data(), imageDataAux.data(), images_number * image_size * sizeof(type));

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

    columns.resize(image_size + 1);

    // Input columns

    Index column_index = 0;

        for(Index i = 0; i < channels; i++)
        {
            for(Index j = 0; j < width; j++)
            {
                for(Index k = 0; k < height ; k++)
                {
                    columns(column_index).name= "pixel_" + to_string(i+1)+ "_" + to_string(j+1) + "_" + to_string(k+1);
                    columns(column_index).type = RawVariableType::Numeric;
                    columns(column_index).column_use = VariableUse::Input;
                    columns(column_index).scaler = Scaler::MinimumMaximum;
                    column_index++;
                }
            }
        }

    // Target columns

    columns(image_size).name = "class";

    if(classes_number == 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_bmp() method.\n"
               << "Invalid number of categories. The minimum is 2 and you have 1.\n";

        throw invalid_argument(buffer.str());


    }

    Tensor<string, 1> categories(classes_number);

    for(Index i = 0 ; i < classes_number; i++)
    {
        categories(i) = classes_folder[i];
    }

    columns(image_size).column_use = VariableUse::Target;
    columns(image_size).categories = categories;

    columns(image_size).categories_uses.resize(classes_number);
    columns(image_size).categories_uses.setConstant(VariableUse::Target);

    if(classes_number == 2)
    {
        columns(image_size).type = RawVariableType::Binary;
    }
    else
    {
        columns(image_size).type = RawVariableType::Categorical;
    }

    samples_uses.resize(images_number);
    split_samples_random();

    image_width = width;
    image_height = height;
    channels_number = channels;

    input_variables_dimensions.resize(3);
    input_variables_dimensions.setValues({height, width, channels});
}


BoundingBox ImageDataSet::propose_random_region(const Tensor<unsigned char, 1>& image) const
{

    const Index channels_number = get_channels_number();
    const Index image_height = get_image_height();
    const Index image_width = get_image_width();

    Index x_center = rand() % image_width;
    Index y_center = rand() % image_height;

    Index x_top_left;
    Index y_top_left;

    if(x_center == 0){x_top_left = 0;}
    else{x_top_left = rand() % x_center;}

    if(y_center == 0){y_top_left = 0;}
    else{y_top_left = rand() % y_center;}

    Index x_bottom_right;

    if(x_top_left == 0){x_bottom_right = rand()%(image_width - (x_center + 1) + 1) + (x_center + 1);}
    else{x_bottom_right = rand()%(image_width - x_center + 1) + x_center;}

    Index y_bottom_right;

    if(y_top_left == 0){y_bottom_right = rand()%(image_height - (y_center + 1) + 1) + (y_center + 1);}
    else{y_bottom_right = rand() % (image_height - y_center + 1) + y_center;}

    BoundingBox random_region(channels_number, x_top_left, y_top_left, x_bottom_right, y_bottom_right);

    random_region.data = get_bounding_box(image, x_top_left, y_top_left, x_bottom_right, y_bottom_right);

    return random_region;
}


void ImageDataSet::read_ground_truth()
{
/*
    srand(time(NULL));

    const Index classes_number = get_label_classes_number_from_XML(data_source_path);
//    const Index annotations_number = get_bounding_boxes_number_from_XML(data_source_path); // This function also save channels, width and height

    const Index target_variables_number = classes_number == 1 ? 1 : classes_number + 1; // 1 class for background

    const Index samples_number = images_number*regions_number;
    const Index variables_number = channels_number*region_rows*region_columns + target_variables_number;

    const Index pixels_number = channels_number*region_rows*region_columns;

    set(samples_number, variables_number);

    data.setZero();

    rows_labels.resize(samples_number);

    Index row_index = 0;

    // Load XML from string

    tinyxml2::XMLDocument document;

    if(document.LoadFile(data_source_path.c_str()))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void load(const string&) method.\n"
               << "Cannot load XML file " << data_source_path << ".\n";

        throw invalid_argument(buffer.str());
    }

    // Read ground Truth XML

    ostringstream buffer;

    const tinyxml2::XMLElement* neural_labeler_element = document.FirstChildElement("NeuralLabeler");

    if(!neural_labeler_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_ground_truth(const tinyxml2::XMLDocument&) method.\n"
               << "NeuralLabeler element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    // Images

    const tinyxml2::XMLElement* images_element = neural_labeler_element -> FirstChildElement("Images");

    if(!images_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_ground_truth(const tinyxml2::XMLDocument&) method.\n"
               << "Images element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    // Images Number

    const tinyxml2::XMLElement* images_number_element = images_element -> FirstChildElement("ImagesNumber");

    if(!images_number_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_ground_truth(const tinyxml2::XMLDocument&) method.\n"
               << "ImagesNumber element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const Index images_number = Index(atoi(images_number_element->GetText()));

    const tinyxml2::XMLElement* start_images_element = images_number_element;

    for(Index image_index = 0; image_index < images_number; image_index++)
    {
        // Image

        const tinyxml2::XMLElement* image_element = start_images_element->NextSiblingElement("Image");
        start_images_element = image_element;

        if(!image_element)
        {
            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void read_ground_truth(const tinyxml2::XMLDocument&) method.\n"
                   << "Image element is nullptr.\n";

            throw invalid_argument(buffer.str());
        }

        // Filename

        const tinyxml2::XMLElement* file_name_element = image_element->FirstChildElement("Filename");

        if(!file_name_element)
        {
            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void read_ground_truth(const tinyxml2::XMLDocument&) method.\n"
                   << "Filename element is nullptr.\n";

            throw invalid_argument(buffer.str());
        }

        const string image_filename = file_name_element->GetText();

        const Tensor<unsigned char, 1> image_pixel_values = read_bmp_image(image_filename);

        // Annotations Number

        const tinyxml2::XMLElement* annotations_number_element = image_element->FirstChildElement("AnnotationsNumber");

        if(!annotations_number_element)
        {
            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void read_ground_truth(const tinyxml2::XMLDocument&) method.\n"
                   << "AnnotationsNumber element is nullptr.\n";

            throw invalid_argument(buffer.str());
        }

        const Index annotations_number = Index(atoi(annotations_number_element->GetText()));

        const tinyxml2::XMLElement* start_annotations_element = annotations_number_element;

        for(Index j = 0; j < annotations_number; j++)
        {
            // Annotation

            const tinyxml2::XMLElement* annotation_element = start_annotations_element->NextSiblingElement("Annotation");

            start_annotations_element = annotation_element;

            if(!annotation_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void read_ground_truth(const tinyxml2::XMLDocument&) method.\n"
                       << "Annotation element is nullptr.\n";

                throw invalid_argument(buffer.str());
            }

            // Label

            const tinyxml2::XMLElement* label_element = annotation_element->FirstChildElement("Label");

            if(!label_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void read_ground_truth(const tinyxml2::XMLDocument&) method.\n"
                       << "Label element is nullptr.\n";

                throw invalid_argument(buffer.str());
            }

            const string ground_truth_class = label_element->GetText();

            // Points

            const tinyxml2::XMLElement* points_element = annotation_element->FirstChildElement("Points");

            if(!points_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void read_ground_truth(const tinyxml2::XMLDocument&) method.\n"
                       << "Points element is nullptr.\n";

                throw invalid_argument(buffer.str());
            }

            const string bounding_box_points = points_element->GetText();

            const Tensor<string, 1> splitted_points = get_tokens(bounding_box_points, ',');

            const int x_top_left = static_cast<int>(stoi(splitted_points[0]));
            const int y_top_left = static_cast<int>(stoi(splitted_points[1]));
            const int x_bottom_right = static_cast<int>(stoi(splitted_points[2]));
            const int y_bottom_right = static_cast<int>(stoi(splitted_points[3]));

            // Segmentation for region proposal (Not implemented, we use random region proposal)

            const BoundingBox ground_truth_bounding_box = BoundingBox(channels_number, x_top_left, y_top_left, x_bottom_right, y_bottom_right);

            Tensor<BoundingBox, 1> random_bounding_box(regions_number);

            for(Index region_index = 0; region_index < regions_number; region_index++)
            {
                random_bounding_box(region_index) = propose_random_region(image_pixel_values);

                const type intersection_over_union = calculate_intersection_over_union(ground_truth_bounding_box, random_bounding_box(region_index));

                if(intersection_over_union >= 0.3) // If IoU > 0.3 -> object class
                {
                    for(Index p = 1; p < labels_tokens.size() + 1; p++)
                    {
                        if(labels_tokens(p - 1) == ground_truth_class)
                        {
                            if(classes_number == 1)
                            {
                                data(row_index, pixels_number) = 1;
                            }
                            else
                            {
                                data(row_index, pixels_number + p) = 1;
                            }
                        }
                    }
                }
                else // If IoU < 0.3 -> background class
                {
                    if(classes_number == 1)
                    {
                        data(row_index, pixels_number) = type(0);
                    }
                    else
                    {
                        data(row_index, pixels_number) = 1;
                    }
                }

                const BoundingBox warped_bounding_box = random_bounding_box(region_index).resize(channels_number, region_rows, region_columns);

                for(Index j = 0; j < pixels_number; j++)
                {
                    data(row_index, j) = warped_bounding_box.data(j);
                }

                rows_labels(row_index) = image_filename;

                row_index++;
            }
        }
    }

    // Input columns

    Index column_index = 0;

    for(Index i = 0; i < channels_number; i++)
    {
        for(Index j = 0; j < region_rows; j++)
        {
            for(Index k = 0; k < region_columns ; k++)
            {
                columns(column_index).name= "pixel_" + to_string(i+1)+ "_" + to_string(j+1) + "_" + to_string(k+1);
                columns(column_index).type = ColumnType::Numeric;
                columns(column_index).column_use = VariableUse::Input;
                columns(column_index).scaler = Scaler::MinimumMaximum;
                column_index++;
            }
        }
    }

    // Target columns

    columns(pixels_number).name = "label";

    if(classes_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_ground_truth() method.\n"
               << "Invalid number of categories. The minimum is 1 and you have 0.\n";

        throw invalid_argument(buffer.str());

    }
    else if(classes_number == 1) // Just one because we include background (1+1)
    {
        columns(pixels_number).column_use = VariableUse::Target;
        columns(pixels_number).type = ColumnType::Binary;
        columns(pixels_number).categories = labels_tokens;

        columns(pixels_number).categories_uses.resize(target_variables_number); // classes_number + Background
        columns(pixels_number).categories_uses.setConstant(VariableUse::Target);
    }
    else
    {
        Tensor<string, 1> categories(target_variables_number);

        columns(pixels_number).column_use = VariableUse::Target;
        columns(pixels_number).type = ColumnType::Categorical;
        columns(pixels_number).categories = labels_tokens;

        columns(pixels_number).categories_uses.resize(target_variables_number); // classes_number + Background
        columns(pixels_number).categories_uses.setConstant(VariableUse::Target);
    }

    split_samples_random();

    input_variables_dimensions.resize(3);
    input_variables_dimensions.setValues({region_rows, region_columns, channels_number});
*/
}


Index ImageDataSet::get_bounding_boxes_number_from_XML(const string& file_name)
{
    Index bounding_boxes_number = 0;
    string image_filename;

    //------------------------------ Load XML from string ------------------------------

    tinyxml2::XMLDocument document;

    if(document.LoadFile(file_name.c_str()))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void load(const string&) method.\n"
               << "Cannot load XML file " << file_name << ".\n";

        throw invalid_argument(buffer.str());
    }

    //--------------------------- Read ground Truth XML---------------------------

    ostringstream buffer;

    const tinyxml2::XMLElement* neural_labeler_element = document.FirstChildElement("NeuralLabeler");

    if(!neural_labeler_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void get_bounding_boxes_number_from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "NeuralLabeler element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    // Images

    const tinyxml2::XMLElement* images_element = neural_labeler_element -> FirstChildElement("Images");

    if(!images_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void get_bounding_boxes_number_from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Images element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    // Images Number

    const tinyxml2::XMLElement* images_number_element = images_element -> FirstChildElement("ImagesNumber");

    if(!images_number_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void get_bounding_boxes_number_from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "ImagesNumber element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const Index images_number = Index(atoi(images_number_element->GetText()));

    set_images_number(images_number);

    const tinyxml2::XMLElement* start_image_element = images_number_element;

    for(Index i = 0; i < images_number; i++)
    {
        // Image

        const tinyxml2::XMLElement* image_element = start_image_element->NextSiblingElement("Image");
        start_image_element = image_element;

        if(!image_element)
        {
            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void get_bounding_boxes_number_from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Image element is nullptr.\n";

            throw invalid_argument(buffer.str());
        }

        // Filename

        const tinyxml2::XMLElement* file_name_element = image_element->FirstChildElement("Filename");

        if(!file_name_element)
        {
            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void get_bounding_boxes_number_from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Filename element is nullptr.\n";

            throw invalid_argument(buffer.str());
        }

        image_filename = file_name_element->GetText();

        // Annotations Number

        const tinyxml2::XMLElement* annotations_number_element = image_element->FirstChildElement("AnnotationsNumber");

        if(!annotations_number_element)
        {
            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void get_bounding_boxes_number_from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "AnnotationsNumber element is nullptr.\n";

            throw invalid_argument(buffer.str());
        }

        const Index annotations_number = Index(atoi(annotations_number_element->GetText()));

        const tinyxml2::XMLElement* start_annotations_element = annotations_number_element;

        for(Index j = 0; j < annotations_number; j++)
        {
            // Annotation

            const tinyxml2::XMLElement* annotation_element = start_annotations_element->NextSiblingElement("Annotation");
            start_annotations_element = annotation_element;

            if(!annotation_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void get_bounding_boxes_number_from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Annotation element is nullptr.\n";

                throw invalid_argument(buffer.str());
            }

            // Label

            const tinyxml2::XMLElement* label_element = annotation_element->FirstChildElement("Label");

            if(!label_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void get_bounding_boxes_number_from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Label element is nullptr.\n";

                throw invalid_argument(buffer.str());
            }

            string ground_truth_class = label_element->GetText();

            // Points

            const tinyxml2::XMLElement* points_element = annotation_element->FirstChildElement("Points");

            if(!points_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void get_bounding_boxes_number_from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Points element is nullptr.\n";

                throw invalid_argument(buffer.str());
            }
        }

        bounding_boxes_number += annotations_number;
    }

    read_bmp_image(image_filename); // Read an image to save initially the channels number

    return bounding_boxes_number;
}


Index ImageDataSet::get_label_classes_number_from_XML(const string& file_name)
{
    //-----------------------Load XML from string-----------------------------------

    tinyxml2::XMLDocument document;

    if(document.LoadFile(file_name.c_str()))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void load(const string&) method.\n"
               << "Cannot load XML file " << file_name << ".\n";

        throw invalid_argument(buffer.str());
    }

    //--------------------------------Read ground Truth XML-------------------------------------/

    ostringstream buffer;

    const tinyxml2::XMLElement* neural_labeler_element = document.FirstChildElement("NeuralLabeler");

    if(!neural_labeler_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void get_label_classes_number_from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "NeuralLabeler element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    // Images

    const tinyxml2::XMLElement* images_element = neural_labeler_element -> FirstChildElement("Images");

    if(!images_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void get_label_classes_number_from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Images element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    // Labels

    const tinyxml2::XMLElement* labels_element = neural_labeler_element -> FirstChildElement("Labels");

    if(!labels_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void get_label_classes_number_from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Labels element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    // Labels Number

    const tinyxml2::XMLElement* labels_number_element = labels_element -> FirstChildElement("LabelsNumber");

    if(!labels_number_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void get_label_classes_number_from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "LabelsNumber element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    if(labels_number_element->GetText())
    {
        const Index labels_number = Index(atoi(labels_number_element->GetText()));

        set_categories_number(labels_number);
    }

    labels_tokens.resize(categories_number);

    const tinyxml2::XMLElement* start_label_element = labels_number_element;

    for(Index i = 0; i < categories_number; i++)
    {
        // Label

        const tinyxml2::XMLElement* label_element = start_label_element->NextSiblingElement("Label");
        start_label_element = label_element;

        if(!label_element)
        {
            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void get_label_classes_number_from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Label element is nullptr.\n";

            throw invalid_argument(buffer.str());
        }

        // Name

        const tinyxml2::XMLElement* name_element = label_element->FirstChildElement("Name");

        if(!name_element)
        {
            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void get_label_classes_number_from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Name element is nullptr.\n";

            throw invalid_argument(buffer.str());
        }

        const string label_string = name_element->GetText();

        labels_tokens(i) = label_string;

        // Color

        const tinyxml2::XMLElement* color_element = label_element->FirstChildElement("Color");

        if(!color_element)
        {
            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void get_label_classes_number_from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Color element is nullptr.\n";

            throw invalid_argument(buffer.str());
        }
    }

    return categories_number;
}


Tensor<type, 1> ImageDataSet::get_bounding_box(const Tensor<unsigned char, 1>& image,
                                          const Index& x_top_left, const Index& y_top_left,
                                          const Index& x_bottom_right, const Index& y_bottom_right) const
{
    const Index channels_number = get_channels_number();
    const Index height = get_image_height();
    const Index width = get_image_width();
    const Index image_size_single_channel = height * width;

    const Index bounding_box_width = abs(x_top_left - x_bottom_right);
    const Index bounding_box_height = abs(y_top_left - y_bottom_right);
    const Index bounding_box_single_channel_size = bounding_box_width * bounding_box_height;

    Tensor<type, 1> bounding_box_data;
    bounding_box_data.resize(channels_number * bounding_box_single_channel_size);

    const Index pixel_loop_start = width * (height - y_bottom_right) + x_top_left;
    const Index pixel_loop_end = width * (height - 1 - y_top_left) + x_bottom_right;

    if(channels_number == 3)
    {
        Tensor<unsigned char, 1> image_red_channel_flatted_sorted(image_size_single_channel);
        Tensor<unsigned char, 1> image_green_channel_flatted_sorted(image_size_single_channel);
        Tensor<unsigned char, 1> image_blue_channel_flatted_sorted(image_size_single_channel);

        image_red_channel_flatted_sorted = image.slice(Eigen::array<Eigen::Index, 1>({0}), Eigen::array<Eigen::Index, 1>({image_size_single_channel}));
        image_green_channel_flatted_sorted = image.slice(Eigen::array<Eigen::Index, 1>({image_size_single_channel}), Eigen::array<Eigen::Index, 1>({image_size_single_channel}));
        image_blue_channel_flatted_sorted = image.slice(Eigen::array<Eigen::Index, 1>({2 * image_size_single_channel}), Eigen::array<Eigen::Index, 1>({image_size_single_channel}));

        Tensor<type, 1> bounding_box_red_channel(bounding_box_single_channel_size);
        Tensor<type, 1> bounding_box_green_channel(bounding_box_single_channel_size);
        Tensor<type, 1> bounding_box_blue_channel(bounding_box_single_channel_size);

        Index data_index = 0;

        for(Index i = pixel_loop_start; i <= pixel_loop_end - 1; i++)
        {
            const int height_number = (int)(i/height);

            const Index left_margin = height_number * width + x_top_left;
            const Index right_margin = height_number * width + x_bottom_right;

            if(i >= left_margin && i < right_margin)
            {
                bounding_box_red_channel(data_index) = type(image_red_channel_flatted_sorted[i]);
                bounding_box_green_channel(data_index) = type(image_green_channel_flatted_sorted[i]);
                bounding_box_blue_channel(data_index) = type(image_blue_channel_flatted_sorted[i]);

                data_index++;
            }
        }

        Tensor<type, 1> red_green_concatenation(bounding_box_red_channel.size() + bounding_box_green_channel.size());
        red_green_concatenation = bounding_box_red_channel.concatenate(bounding_box_green_channel, 0); // To allow a double concatenation
        bounding_box_data = red_green_concatenation.concatenate(bounding_box_blue_channel, 0);
    }
    else
    {
        Index data_index = 0;

        for(Index i = pixel_loop_start; i <= pixel_loop_end - 1; i++)
        {
            const int height_number = (int)(i/height);

            const Index left_margin = height_number * width + x_top_left;
            const Index right_margin = height_number * width + x_bottom_right;

            if(i >= left_margin && i < right_margin)
            {
                bounding_box_data(data_index) = type(image[i]);
                data_index++;
            }
        }
    }

    return bounding_box_data;
}


void ImageDataSet::set_categories_number(const Index& new_categories_number)
{
    categories_number = new_categories_number;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2023 Artificial Intelligence Techniques, SL.
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
