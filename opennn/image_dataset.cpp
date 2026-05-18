//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I M A G E   D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "image_dataset.h"
#include "image_utilities.h"
#include "tensor_utilities.h"
#include "string_utilities.h"
#include "random_utilities.h"

namespace opennn
{

ImageDataset::ImageDataset(const Index new_samples_number,
                           const Shape& new_input_shape,
                           const Shape& new_target_shape)
{
    if (new_input_shape.size() != 3)
        throw runtime_error("Input shape is not 3");

    if (new_target_shape.size() != 1)
        throw runtime_error("Target shape is not 1");

    set(new_samples_number, new_input_shape, new_target_shape);
}


ImageDataset::ImageDataset(const filesystem::path& new_data_path) : Dataset()
{
    data_path = new_data_path;

    read_bmp();
}


Index ImageDataset::get_channels_number() const
{
    return input_shape[2];
}


Index ImageDataset::get_image_width() const
{
    return input_shape[1];
}


Index ImageDataset::get_image_height() const
{
    return input_shape[0];
}


Index ImageDataset::get_image_size() const
{
    return input_shape[0] * input_shape[1] * input_shape[2];
}


Index ImageDataset::get_image_padding() const
{
    return padding;
}


bool ImageDataset::get_augmentation() const
{
    return augmentation;
}


bool ImageDataset::get_random_reflection_axis_x() const
{
    return random_reflection_axis_x;
}


bool ImageDataset::get_random_reflection_axis_y() const
{
    return random_reflection_axis_y;
}


type ImageDataset::get_random_rotation_minimum() const
{
    return random_rotation_minimum;
}


type ImageDataset::get_random_rotation_maximum() const
{
    return random_rotation_maximum;
}


type ImageDataset::get_random_horizontal_translation_minimum() const
{
    return random_horizontal_translation_minimum;
}


type ImageDataset::get_random_horizontal_translation_maximum() const
{
    return random_horizontal_translation_maximum;
}


type ImageDataset::get_random_vertical_translation_maximum() const
{
    return random_vertical_translation_maximum;
}


type ImageDataset::get_random_vertical_translation_minimum() const
{
    return random_vertical_translation_minimum;
}


void ImageDataset::set_data_random()
{
    const Index height = input_shape[0];
    const Index width = input_shape[1];
    const Index channels = input_shape[2];

    const Index targets_number = target_shape[0];
    const Index inputs_number = height * width * channels;
    const Index samples_number = data.rows();

    data.setZero();

    if (targets_number == 1)
    {
        const Index half_samples = samples_number / 2;

        for(Index i = 0; i < samples_number; i++)
        {
            for(Index j = 0; j < inputs_number; j++)
                data(i, j) = random_integer(0, 255);

            data(i, inputs_number) = (i < half_samples) ? 0 : 1;
        }
    }
    else
    {
        VectorI images_number(targets_number);
        images_number.setZero();

        const Index images_per_category = samples_number / targets_number;
        Index remainder = samples_number % targets_number;

        for(Index i = 0; i < targets_number; i++)
        {
            images_number[i] = images_per_category + (remainder > 0 ? 1 : 0);

            if (remainder > 0)
                remainder--;
        }

        Index current_sample = 0;

        for(Index k = 0; k < targets_number; k++)
        {
            for(Index i = 0; i < images_number[k]; i++)
            {
                for(Index j = 0; j < inputs_number; j++)
                    data(current_sample, j) = random_integer(0, 255);

                data(current_sample, k + inputs_number) = 1;

                current_sample++;
            }
        }
    }
}


void ImageDataset::set_channels_number(const int& new_channels)
{
    input_shape[2] = new_channels;
}


void ImageDataset::set_image_width(const int& new_width)
{
    input_shape[1] = new_width;
}


void ImageDataset::set_image_height(const int& new_height)
{
    input_shape[0] = new_height;
}


void ImageDataset::set_image_padding(const int& new_padding)
{
    padding = new_padding;
}


void ImageDataset::set_augmentation(bool new_augmentation)
{
    augmentation = new_augmentation;
}


void ImageDataset::set_random_reflection_axis_x(bool new_random_reflection_axis_x)
{
    random_reflection_axis_x = new_random_reflection_axis_x;
}


void ImageDataset::set_random_reflection_axis_y(bool new_random_reflection_axis_y)
{
    random_reflection_axis_y = new_random_reflection_axis_y;
}


void ImageDataset::set_random_rotation_minimum(const type new_random_rotation_minimum)
{
    random_rotation_minimum = new_random_rotation_minimum;
}


void ImageDataset::set_random_rotation_maximum(const type new_random_rotation_maximum)
{
    random_rotation_maximum = new_random_rotation_maximum;
}


void ImageDataset::set_random_horizontal_translation_maximum(const type new_random_horizontal_translation_maximum)
{
    random_horizontal_translation_maximum = new_random_horizontal_translation_maximum;
}


void ImageDataset::set_random_horizontal_translation_minimum(const type new_random_horizontal_translation_minimum)
{
    random_horizontal_translation_minimum = new_random_horizontal_translation_minimum;
}


void ImageDataset::set_random_vertical_translation_minimum(const type new_random_vertical_translation_minimum)
{
    random_vertical_translation_minimum = new_random_vertical_translation_minimum;
}


void ImageDataset::set_random_vertical_translation_maximum(const type new_random_vertical_translation_maximum)
{
    random_vertical_translation_maximum = new_random_vertical_translation_maximum;
}


void ImageDataset::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("ImageDataset");

    printer.OpenElement("DataSource");

    add_xml_element(printer, "FileType", "bmp");
    add_xml_element(printer, "Path", data_path.string());
    add_xml_element(printer, "HasSamplesId", to_string(has_sample_ids));
    add_xml_element(printer, "Channels", to_string(get_channels_number()));
    add_xml_element(printer, "Width", to_string(get_image_width()));
    add_xml_element(printer, "Height", to_string(get_image_height()));
    add_xml_element(printer, "Padding", to_string(get_image_padding()));
    add_xml_element(printer, "RandomReflectionAxisX", to_string(get_random_reflection_axis_x()));
    add_xml_element(printer, "RandomReflectionAxisY", to_string(get_random_reflection_axis_y()));
    add_xml_element(printer, "RandomRotationMinimum", to_string(get_random_rotation_minimum()));
    add_xml_element(printer, "RandomRotationMaximum", to_string(get_random_rotation_maximum()));
    add_xml_element(printer, "RandomHorizontalTranslationMinimum", to_string(get_random_horizontal_translation_minimum()));
    add_xml_element(printer, "RandomHorizontalTranslationMaximum", to_string(get_random_horizontal_translation_maximum()));
    add_xml_element(printer, "RandomVerticalTranslationMinimum", to_string(get_random_vertical_translation_minimum()));
    add_xml_element(printer, "RandomVerticalTranslationMaximum", to_string(get_random_vertical_translation_maximum()));
    add_xml_element(printer, "Codification", get_codification_string());

    printer.CloseElement();

    image_variables_to_XML(printer);

    image_samples_to_XML(printer);

    add_xml_element(printer, "Display", to_string(display));

    printer.CloseElement();
}


void ImageDataset::set_default_variable_names()
{
    // Pixel inputs in an image dataset do not need per-variable names: they
    // never surface in the UI, reports or expressions. The 270k-string init
    // loop in the base implementation is pure overhead for image classification.
}


void ImageDataset::image_samples_to_XML(XMLPrinter& printer) const
{
    // Base Dataset::samples_to_XML uses get_samples_number() — which returns
    // data.rows(). For image-classification tasks that never call
    // load_data_binary (e.g., Report dataset, Randomize parameters), data is
    // empty and that would emit <SamplesNumber>0</SamplesNumber>, corrupting
    // the .ndm. sample_roles is the source of truth, so we use it for both
    // <SamplesNumber> and <SampleRoles> (the base get_sample_roles_vector()
    // also iterates from 0 to get_samples_number() and would return an
    // empty vector here).

    printer.OpenElement("Samples");

    const Index roles_count = Index(sample_roles.size());
    add_xml_element(printer, "SamplesNumber", to_string(roles_count));

    if(has_sample_ids)
        add_xml_element(printer, "SamplesId", vector_to_string(sample_ids, get_separator_string()));

    // Build the "0 1 2 3"-style roles string straight from sample_roles to
    // avoid the data.rows() trap of get_sample_roles_vector().
    string roles_str;
    roles_str.reserve(roles_count * 2);
    for(Index i = 0; i < roles_count; ++i)
    {
        if(i > 0) roles_str.push_back(' ');
        const string& r = sample_roles[i];
        if(r == "Training")        roles_str.push_back('0');
        else if(r == "Validation") roles_str.push_back('1');
        else if(r == "Testing")    roles_str.push_back('2');
        else                       roles_str.push_back('3'); // "None" or unset
    }
    add_xml_element(printer, "SampleRoles", roles_str);

    printer.CloseElement();
}


void ImageDataset::samples_from_XML(const XMLElement* samples_element)
{
    // Base Dataset::samples_from_XML resizes and zero-fills the `data` matrix
    // to (samples_number × variables_number). For a 300x300x3 / 10k-sample image
    // project this is ~11 GB just for the zero-init — the bulk of fromXML time
    // for any post-import task. The image pixel data lives in the binary cache
    // and is only materialised when a task explicitly calls load_data_binary
    // (which does its own resize). So here we record metadata and give `data`
    // the right row count with zero columns — that way `get_samples_number()`
    // (which returns data.rows()) reports the correct value for tasks that
    // do not load the binary (e.g., Report dataset), without paying the
    // 11 GB allocation.

    if(!samples_element)
        throw runtime_error("Samples element is nullptr.\n");

    const Index samples_number = read_xml_index(samples_element, "SamplesNumber");

    if(has_sample_ids)
    {
        const string separator_string = get_separator_string();
        sample_ids = get_tokens(read_xml_string(samples_element, "SamplesId"), separator_string);
    }

    sample_roles.resize(samples_number);

    if(samples_number == 0)
        return;

    // Filter empty tokens from the roles list (the XML text can have trailing
    // whitespace, and an empty <SampleRoles/> element returns [""] from
    // get_tokens). set_sample_roles would throw on the empty string.
    const vector<string> raw_tokens = get_tokens(read_xml_string(samples_element, "SampleRoles"), " ");
    vector<string> role_tokens;
    role_tokens.reserve(raw_tokens.size());
    for(const string& token : raw_tokens)
        if(!token.empty())
            role_tokens.push_back(token);

    if(!role_tokens.empty())
        set_sample_roles(role_tokens);
}


void ImageDataset::image_variables_to_XML(XMLPrinter& printer) const
{
    const Index total_variables = get_variables_number();
    const Index input_pixels_count = get_image_size();

    printer.OpenElement("Variables");
    add_xml_element(printer, "VariablesNumber", to_string(total_variables));
    add_xml_element(printer, "InputVariablesNumber", to_string(input_pixels_count));

    // Pixel inputs are implicit (reconstructed at load time with defaults).
    // Emit only target/unused variables in full.
    for(Index i = input_pixels_count; i < total_variables; i++)
    {
        printer.OpenElement("Variable");
        printer.PushAttribute("Item", to_string(i + 1).c_str());
        variables[i].to_XML(printer);
        printer.CloseElement();
    }

    printer.CloseElement();
}


void ImageDataset::image_variables_from_XML(const XMLElement* variables_element)
{
    if(!variables_element)
        throw runtime_error("Variables element is nullptr.\n");

    const Index variables_number = read_xml_index(variables_element, "VariablesNumber");
    const Index input_pixels_count = read_xml_index(variables_element, "InputVariablesNumber");

    set_variables_number(variables_number);

    // Pixel inputs are decorative metadata (only role matters downstream filters).
    // Default-constructed Variables already have type=Numeric, scaler="MeanStandardDeviation",
    // empty name and categories — only the role needs to be set to "Input".
    static const string input_role = "Input";
    for(Index i = 0; i < input_pixels_count; i++)
        variables[i].role = input_role;

    const XMLElement* variable_element = variables_element->FirstChildElement("Variable");

    for(Index i = input_pixels_count; i < variables_number; i++)
    {
        if(!variable_element)
            throw runtime_error("Missing <Variable> entry for index " + to_string(i + 1) + ".\n");

        const char* item_attr = variable_element->Attribute("Item");
        if(!item_attr || to_string(i + 1) != item_attr)
            throw runtime_error("Variable item mismatch at index " + to_string(i + 1) + ".\n");

        Variable& variable = variables[i];
        variable.name = read_xml_string(variable_element, "Name");
        variable.set_scaler(read_xml_string(variable_element, "Scaler"));
        variable.set_role(read_xml_string(variable_element, "Role"));
        variable.set_type(read_xml_string(variable_element, "Type"));

        if(variable.type == VariableType::Categorical || variable.type == VariableType::Binary)
        {
            const XMLElement* categories_element = variable_element->FirstChildElement("Categories");

            if(categories_element)
                variable.categories = get_tokens(read_xml_string(variable_element, "Categories"), ";");
            else if(variable.type == VariableType::Binary)
                variable.categories = { "0", "1" };
            else
                throw runtime_error("Categorical Variable Element is nullptr: Categories");
        }

        variable_element = variable_element->NextSiblingElement("Variable");
    }
}


void ImageDataset::print() const
{
    Dataset::print();
}


void ImageDataset::perform_augmentation(type* input_data) const
{
    throw runtime_error("Image Augmentation is not yet implemented. Please check back in a future version.");

    const Shape input_shape = get_shape("Input");

    const Index samples_number = input_shape[0];
    const Index input_height = input_shape[0];
    const Index input_width = input_shape[1];
    const Index channels = input_shape[2];

    TensorMap4 inputs(input_data,
                      samples_number,
                      input_height,
                      input_width,
                      channels);

    for(Index batch_index = 0; batch_index < samples_number; batch_index++)
    {
        Tensor3 image = inputs.chip(batch_index, 0);

        if(random_reflection_axis_x)
            reflect_image_x(image);

        if(random_reflection_axis_y)
            reflect_image_y(image);

        if(random_rotation_minimum != 0 && random_rotation_maximum != 0)
            rotate_image(image, image, random_uniform(random_rotation_minimum, random_rotation_maximum));

        if(random_horizontal_translation_minimum != 0 && random_horizontal_translation_maximum != 0)
            translate_image_x(image, image, random_uniform(random_horizontal_translation_minimum, random_horizontal_translation_maximum));

        if(random_vertical_translation_minimum != 0 && random_vertical_translation_maximum != 0)
            translate_image_y(image, image, random_uniform(random_vertical_translation_minimum, random_vertical_translation_maximum));
    }
}


void ImageDataset::fill_inputs(const vector<Index>& sample_indices, const vector<Index>& input_indices, type* input_data, bool parallelize) const
{
    fill_tensor_data(data, sample_indices, input_indices, input_data, parallelize);

    if (augmentation)
        perform_augmentation(input_data);

}


void ImageDataset::from_XML(const XMLDocument& data_set_document)
{
    using __clkD = chrono::high_resolution_clock;
    auto __msD = [](auto a, auto b) {
        return chrono::duration_cast<chrono::milliseconds>(b - a).count();
    };
    auto __sD = __clkD::now();

    const XMLElement* image_dataset_element = data_set_document.FirstChildElement("ImageDataset");

    if(!image_dataset_element)
        throw runtime_error("ImageDataset element is nullptr.\n");

    // Data Source

    const XMLElement* data_source_element = image_dataset_element->FirstChildElement("DataSource");

    if(!data_source_element)
        throw runtime_error("Element is nullptr: DataSource");

    set_data_path(read_xml_string(data_source_element, "Path"));
    set_has_ids(read_xml_bool(data_source_element, "HasSamplesId"));

    __sD = __clkD::now();
    set_shape("Input", { read_xml_index(data_source_element, "Height"),
                         read_xml_index(data_source_element, "Width"),
                         read_xml_index(data_source_element, "Channels") });
    cout << "[TIMING-ENG]       img.set_shape: " << __msD(__sD, __clkD::now()) << " ms" << endl;

    set_image_padding(read_xml_index(data_source_element, "Padding"));

    set_codification(read_xml_string(data_source_element, "Codification"));

    set_random_reflection_axis_x(read_xml_index(data_source_element, "RandomReflectionAxisX"));
    set_random_reflection_axis_y(read_xml_index(data_source_element, "RandomReflectionAxisY"));
    set_random_rotation_minimum(type(atof(read_xml_string(data_source_element, "RandomRotationMinimum").c_str())));
    set_random_rotation_maximum(type(atof(read_xml_string(data_source_element, "RandomRotationMaximum").c_str())));
    set_random_horizontal_translation_minimum(type(atof(read_xml_string(data_source_element, "RandomHorizontalTranslationMinimum").c_str())));
    set_random_horizontal_translation_maximum(type(atof(read_xml_string(data_source_element, "RandomHorizontalTranslationMaximum").c_str())));
    set_random_vertical_translation_minimum(type(atof(read_xml_string(data_source_element, "RandomVerticalTranslationMinimum").c_str())));
    set_random_vertical_translation_maximum(type(atof(read_xml_string(data_source_element, "RandomVerticalTranslationMaximum").c_str())));

    // Variables

    const XMLElement* variables_element = image_dataset_element->FirstChildElement("Variables");

    __sD = __clkD::now();
    image_variables_from_XML(variables_element);
    cout << "[TIMING-ENG]       img.variables_from_XML: " << __msD(__sD, __clkD::now()) << " ms" << endl;

    // Samples

    const XMLElement* samples_element = image_dataset_element->FirstChildElement("Samples");

    __sD = __clkD::now();
    samples_from_XML(samples_element);
    cout << "[TIMING-ENG]       img.samples_from_XML: " << __msD(__sD, __clkD::now()) << " ms" << endl;
}


vector<Descriptives> ImageDataset::scale_features(const string&)
{
    const Index samples_number = get_samples_number();
    const Index input_features_number = get_features_number("Input");

    #pragma omp parallel for
    for(Index i = 0; i < samples_number; i++)
        for(Index j = 0; j < input_features_number; j++)
            data(i, j) /= type(255);

    return {};
}


void ImageDataset::unscale_features(const string&)
{
    const Index samples_number = get_samples_number();
    const Index input_features_number = get_features_number("Input");

    #pragma omp parallel for
    for(Index i = 0; i < samples_number; i++)
        for(Index j = 0; j < input_features_number; j++)
            data(i, j) *= type(255);
}


void ImageDataset::read_bmp(const Shape& new_input_shape)
{
    chrono::high_resolution_clock::time_point start_time = chrono::high_resolution_clock::now();
    
    vector<filesystem::path> directory_path;

    for(const filesystem::directory_entry& current_directory : filesystem::directory_iterator(data_path))
        if(current_directory.is_directory())
            directory_path.emplace_back(current_directory.path());

    const Index folders_number = directory_path.size();

    VectorI images_number(folders_number + 1);
    images_number.setZero();
    
    Index samples_number = 0;

    vector<string> image_path;

    for(Index i = 0; i < folders_number; i++)
    {
        for(const filesystem::directory_entry& current_directory : filesystem::directory_iterator(directory_path[i]))
        {
            if(current_directory.is_regular_file() && current_directory.path().extension() == ".bmp")
            {
                image_path.emplace_back(current_directory.path().string());
                samples_number++;
            }
        }

        images_number[i+1] = samples_number;
    }

    if (samples_number == 0)
        throw runtime_error("No images in folder \n");

    const Tensor3 image = load_image(image_path[0]);

    Index height = image.dimension(0);
    Index width = image.dimension(1);
    const Index channels = image.dimension(2);

    if (new_input_shape[2] != channels && new_input_shape[2] != 0)
        throw runtime_error("Different number of channels in new_input_shape \n");

    if (new_input_shape[0] != 0 && new_input_shape[1] != 0)
    {
        height = new_input_shape[0];
        width = new_input_shape[1];
    }

    const Index pixels_number = height * width * channels;

    const Index targets_number = (folders_number == 2) 
        ? folders_number - 1 
        : folders_number;

    set(samples_number, { height, width, channels }, { targets_number });

    if (targets_number == 1)
    {
        variables[pixels_number].name = directory_path[0].filename().string() + "_" + directory_path[1].filename().string();
        variables[pixels_number].role = "Target";
        variables[pixels_number].type = VariableType::Binary;
        variables[pixels_number].set_categories({directory_path[0].filename().string(), directory_path[1].filename().string()});
        variables[pixels_number].scaler = "None";
    }
    else
    {
        variables.resize(pixels_number + 1); 

        vector<string> categories(targets_number);
        for(Index i = 0; i < targets_number; i++)
            categories[i] = directory_path[i].filename().string();

        variables[pixels_number].name = "Class";
        variables[pixels_number].role = "Target";
        variables[pixels_number].type = VariableType::Categorical;
        variables[pixels_number].set_categories(categories);
        variables[pixels_number].scaler = "None";
    }

    Index progress_counter = 0;

    #pragma omp parallel for
    for(Index i = 0; i < samples_number; i++)
    {
        Tensor3 image = load_image(image_path[i]);

        const Index current_height = image.dimension(0);
        const Index current_width = image.dimension(1);
        const Index current_channels = image.dimension(2);

        if (current_channels != channels)
            throw runtime_error("Different number of channels in image: " + image_path[i] + "\n");

        if (current_height != height || current_width != width)
            image = resize_image(image, height, width);

        copy(image.data(), image.data() + pixels_number, &data(i, 0));

        if (targets_number == 1)
            data(i, pixels_number) = (i >= images_number(0) && i < images_number(1)) ? 0 : 1;
        else
            for(Index k = 0; k < targets_number; k++)
                if (i >= images_number(k) && i < images_number(k + 1))
                    data(i, k + pixels_number) = 1;
                else
                    data(i, k + pixels_number) = 0; 

        #pragma omp atomic
        progress_counter++;

        if (omp_get_thread_num() == 0)
            display_progress_bar(progress_counter, samples_number);
    }

    if (display)
    {
        chrono::high_resolution_clock::time_point end_time = chrono::high_resolution_clock::now();
        chrono::milliseconds duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

        const long long total_milliseconds = duration.count();
        const long long minutes = total_milliseconds / 60000;
        const long long seconds = (total_milliseconds % 60000) / 1000;
        const long long milliseconds = total_milliseconds % 1000;

        cout << "\nImage dataset loaded in: "
             << minutes << " minutes, "
             << seconds << " seconds, "
             << milliseconds << " milliseconds." << endl;
    }

    sample_ids.clear();
    sample_ids.reserve(image_path.size());
    for(const string& path : image_path)
        sample_ids.emplace_back(filesystem::path(path).filename().string());
    has_sample_ids = true;

    for(Index i = samples_number - 1; i > 0; --i)
    {
        const Index j = random_integer(Index(0), i);
        if(i == j) continue;
        data.row(i).swap(data.row(j));
        std::swap(sample_ids[i], sample_ids[j]);
    }
}

} // opennn namespace


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
