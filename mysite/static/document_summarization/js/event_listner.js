document.addEventListener('DOMContentLoaded', function () {
    const fileInput = document.querySelector('input[type="file"]');
    const alertContainer = document.createElement('div');

    alertContainer.className = 'dropdown-alert alert alert-success text-center';
    alertContainer.style.display = 'none';
    alertContainer.style.position = 'fixed';
    alertContainer.style.top = '0';
    alertContainer.style.left = '0';
    alertContainer.style.right = '0';
    alertContainer.style.zIndex = '1050';
    alertContainer.style.borderRadius = '0';
    alertContainer.style.margin = '0';
    alertContainer.style.transition = 'top 0.5s ease';

    document.body.appendChild(alertContainer);

    fileInput.addEventListener('change', function () {
        if (fileInput.files.length > 0) {
            alertContainer.textContent = '📄 "' + fileInput.files[0].name + '" selected!';
            alertContainer.style.display = 'block';

            // Auto-hide after 3 seconds
            setTimeout(() => {
                alertContainer.style.display = 'none';
            }, 3000);
        } else {
            alertContainer.style.display = 'none';
        }
    });
});